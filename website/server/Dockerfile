# Built with the repository root as the build context (see fly.toml at the
# repo root, or pass `-f website/server/Dockerfile .` from the repo root) --
# not from this directory -- because model.js resolves the template bank at
# a path relative to this service's own location inside the monorepo
# (website/server/src/recognition/../../../../recognition/models), so the
# image needs both website/server/ and recognition/models/ copied in with
# that same relative layout.
#
# Debian-based (glibc), not Alpine (musl): onnxruntime-node ships prebuilt
# native bindings that link against glibc and fail to load on musl libc.
FROM node:22-slim

WORKDIR /app/website/server

COPY website/server/package.json website/server/package-lock.json ./
RUN npm ci --omit=dev

COPY website/server/ ./
COPY recognition/models /app/recognition/models

ENV NODE_ENV=production
EXPOSE 4000

CMD ["node", "src/index.js"]
