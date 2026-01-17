#!/usr/bin/env bash
set -euo pipefail

pnpm test:physics
pnpm test:posteriors
pnpm test:reference
