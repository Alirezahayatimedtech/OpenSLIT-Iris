#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

CVAT_VERSION="${CVAT_VERSION:-v2.64.0}"
CVAT_HOST="${CVAT_HOST:-localhost}"
CVAT_REPOSITORY="${CVAT_REPOSITORY:-https://github.com/cvat-ai/cvat.git}"
CVAT_INSTALL_DIR="${CVAT_INSTALL_DIR:-.runtime/cvat}"

if [[ "${CVAT_INSTALL_DIR}" != /* ]]; then
  CVAT_INSTALL_DIR="${REPO_ROOT}/${CVAT_INSTALL_DIR}"
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

check_prerequisites() {
  require_command git
  require_command docker
  if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required: docker compose" >&2
    exit 1
  fi
}

install_cvat() {
  check_prerequisites
  mkdir -p "$(dirname "${CVAT_INSTALL_DIR}")"

  if [[ ! -d "${CVAT_INSTALL_DIR}/.git" ]]; then
    echo "Cloning CVAT ${CVAT_VERSION} into ${CVAT_INSTALL_DIR}"
    git clone --branch "${CVAT_VERSION}" --depth 1 \
      "${CVAT_REPOSITORY}" "${CVAT_INSTALL_DIR}"
  else
    echo "Using existing CVAT checkout at ${CVAT_INSTALL_DIR}"
    git -C "${CVAT_INSTALL_DIR}" fetch --tags --quiet
    if [[ -n "$(git -C "${CVAT_INSTALL_DIR}" status --porcelain)" ]]; then
      echo "CVAT checkout contains local changes; refusing to change versions." >&2
      exit 1
    fi
    git -C "${CVAT_INSTALL_DIR}" checkout --quiet "${CVAT_VERSION}"
  fi

  echo "CVAT runtime ready at ${CVAT_INSTALL_DIR}"
}

compose() {
  install_cvat
  (
    cd "${CVAT_INSTALL_DIR}"
    export CVAT_VERSION CVAT_HOST
    docker compose "$@"
  )
}

usage() {
  cat <<'EOF'
Usage: deployment/cvat/cvat.sh COMMAND

Commands:
  install           Clone and pin the official CVAT release
  up                Start CVAT in the background
  down              Stop CVAT
  restart           Restart CVAT
  status            Show container status
  logs              Follow CVAT logs
  create-superuser  Create the first CVAT administrator interactively
  url               Print the local CVAT URL

Configuration is read from deployment/cvat/.env.
EOF
}

command_name="${1:-}"
case "${command_name}" in
  install)
    install_cvat
    ;;
  up)
    compose up -d
    echo "CVAT: http://${CVAT_HOST}:8080"
    ;;
  down)
    compose down
    ;;
  restart)
    compose down
    compose up -d
    echo "CVAT: http://${CVAT_HOST}:8080"
    ;;
  status)
    compose ps
    ;;
  logs)
    compose logs -f --tail=200
    ;;
  create-superuser)
    compose exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
    ;;
  url)
    echo "http://${CVAT_HOST}:8080"
    ;;
  *)
    usage
    exit 2
    ;;
esac
