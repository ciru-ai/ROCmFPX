import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    content_no_draft = res.body["content"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    })
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    content_draft = res.body["content"]

    assert content_no_draft == content_draft


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_request_level_speculative_params_are_reported_and_clamped():
    global server
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 4
    server.start()

    cases = [
        ({
            "speculative.n_max": 2,
            "speculative.n_min": 1,
            "speculative.p_min": 0.25,
            "speculative.p_split": 0.2,
        }, (2, 1, 0.25, 0.2)),
        ({
            "speculative.n_max": 100,
            "speculative.n_min": 100,
            "speculative.p_min": 2.0,
            "speculative.p_split": 2.0,
        }, (4, 4, 1.0, 1.0)),
        ({
            "speculative.n_max": 0,
            "speculative.n_min": 1,
            "speculative.p_min": -1.0,
            "speculative.p_split": -1.0,
        }, (0, 0, 0.0, 0.0)),
    ]

    for request_params, expected in cases:
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
            **request_params,
        })
        assert res.status_code == 200
        settings = res.body["generation_settings"]
        assert settings["speculative.n_max"] == expected[0]
        assert settings["speculative.n_min"] == expected[1]
        assert settings["speculative.p_min"] == pytest.approx(expected[2])
        assert settings["speculative.p_split"] == pytest.approx(expected[3])


def test_combined_speculator_omitted_request_uses_largest_capacity():
    global server
    server.spec_type = "ngram-mod,draft-simple"
    server.spec_draft_n_max = 4
    server.spec_ngram_mod_n_match = 24
    server.spec_ngram_mod_n_min = 32
    server.spec_ngram_mod_n_max = 32
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    }
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["generation_settings"]["speculative.n_max"] == 32

    res = server.make_request("POST", "/completion", data={
        **request,
        "speculative.n_max": 2,
    })
    assert res.status_code == 200
    assert res.body["generation_settings"]["speculative.n_max"] == 2


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])
