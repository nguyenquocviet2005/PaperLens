#!/bin/sh
# Vercel build: copy the static site into dist/ and render config.js from project
# environment variables (the same substitution docker-entrypoint.sh does for nginx).
set -eu

: "${PAPERLENS_API_URL:?PAPERLENS_API_URL must be set in the Vercel project}"
AUTH_URL="${SUPABASE_AUTH_URL:-}"
ANON_KEY="${SUPABASE_PUBLISHABLE_KEY:-${SUPABASE_ANON_KEY:-}}"

rm -rf dist
mkdir dist
cp -R index.html app.js styles.css vendor dist/

escaped_api="$(printf '%s' "$PAPERLENS_API_URL" | sed 's/[&|]/\\&/g')"
escaped_auth="$(printf '%s' "$AUTH_URL" | sed 's/[&|]/\\&/g')"
escaped_anon="$(printf '%s' "$ANON_KEY" | sed 's/[&|]/\\&/g')"
sed \
  -e "s|\${PAPERLENS_API_URL}|$escaped_api|g" \
  -e "s|\${SUPABASE_AUTH_URL}|$escaped_auth|g" \
  -e "s|\${SUPABASE_ANON_KEY}|$escaped_anon|g" \
  config.template.js \
  > dist/config.js
