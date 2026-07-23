#!/usr/bin/env bash
# Print or install the Laguna ROCmFP4 Vulkan build dependencies for common Linux distributions.

set -euo pipefail

MODE="${1:---print}"
if [[ "$MODE" != "--print" && "$MODE" != "--install" ]]; then
    echo "usage: $0 [--print|--install]" >&2
    exit 2
fi

if [[ ! -r /etc/os-release ]]; then
    echo "cannot detect this Linux distribution: /etc/os-release is missing" >&2
    exit 2
fi

# shellcheck disable=SC1091
source /etc/os-release
DISTRO="${LAGUNA_DISTRO_ID:-${ID:-unknown}}"
DISTRO_LIKE=" ${LAGUNA_DISTRO_LIKE:-${ID_LIKE:-}} "

run_or_print() {
    if [[ "$MODE" == "--print" ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

root_command() {
    if [[ "${EUID:-$(id -u)}" == "0" ]]; then
        run_or_print "$@"
    else
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required for system package installation" >&2
            exit 2
        fi
        run_or_print sudo "$@"
    fi
}

case "$DISTRO" in
    ubuntu|debian|linuxmint|pop)
        root_command apt-get update
        root_command apt-get install -y \
            git cmake ninja-build build-essential glslc \
            libvulkan-dev vulkan-tools spirv-headers mesa-vulkan-drivers
        ;;
    fedora|rhel|centos|rocky|almalinux)
        root_command dnf install -y \
            git cmake ninja-build gcc gcc-c++ glslc \
            vulkan-loader-devel vulkan-headers spirv-headers \
            vulkan-tools mesa-vulkan-drivers
        ;;
    arch|manjaro|endeavouros)
        root_command pacman -S --needed \
            git cmake ninja base-devel shaderc \
            vulkan-icd-loader vulkan-headers spirv-headers \
            vulkan-tools vulkan-radeon
        ;;
    nixos)
        run_or_print nix --extra-experimental-features "nix-command flakes" profile add \
            nixpkgs#git nixpkgs#cmake nixpkgs#ninja nixpkgs#gcc \
            nixpkgs#shaderc nixpkgs#vulkan-headers nixpkgs#vulkan-loader \
            nixpkgs#spirv-headers
        ;;
    *)
        if [[ "$DISTRO_LIKE" == *" debian "* ]]; then
            root_command apt-get update
            root_command apt-get install -y \
                git cmake ninja-build build-essential glslc \
                libvulkan-dev vulkan-tools spirv-headers mesa-vulkan-drivers
        elif [[ "$DISTRO_LIKE" == *" fedora "* || "$DISTRO_LIKE" == *" rhel "* ]]; then
            root_command dnf install -y \
                git cmake ninja-build gcc gcc-c++ glslc \
                vulkan-loader-devel vulkan-headers spirv-headers \
                vulkan-tools mesa-vulkan-drivers
        elif [[ "$DISTRO_LIKE" == *" arch "* ]]; then
            root_command pacman -S --needed \
                git cmake ninja base-devel shaderc \
                vulkan-icd-loader vulkan-headers spirv-headers \
                vulkan-tools vulkan-radeon
        else
            echo "unsupported automatic dependency installer for ID=$DISTRO ID_LIKE=${ID_LIKE:-}" >&2
            echo "Install: git, cmake, ninja, a C/C++ toolchain, glslc, Vulkan headers/loader/tools, SPIR-V headers, and an AMD Vulkan driver." >&2
            exit 2
        fi
        ;;
esac
