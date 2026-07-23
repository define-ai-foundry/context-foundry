#!/bin/sh
# Generate a throwaway CA + server + client certs for the local TLS CoT listener.
# Dev only — mirrors TAK's mutual-TLS stdssl input without the real TAK CA.
# Runs inside the container (openssl lives there, not necessarily on the host).
set -eu
cd "$(dirname "$0")"
mkdir -p certs
cd certs

if [ -f ca.pem ] && [ -f server.pem ] && [ -f client.pem ]; then
    echo "certs already present; delete dev/certs to regenerate"
    exit 0
fi

# CA
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem \
    -subj "/CN=context-foundry-dev-ca" -days 365 2>/dev/null

# Server cert (CN is arbitrary; the client uses check_hostname=False).
openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr \
    -subj "/CN=tls-cot-listener" 2>/dev/null
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -out server.pem -days 365 2>/dev/null

# Client cert for the fusion engine.
openssl req -newkey rsa:2048 -nodes -keyout client.key -out client.csr \
    -subj "/CN=context-foundry-fusion" 2>/dev/null
openssl x509 -req -in client.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -out client.pem -days 365 2>/dev/null

rm -f server.csr client.csr ca.srl
# World-readable so the unprivileged container user can read them (dev only).
chmod 644 ./*.key ./*.pem
echo "generated dev certs in dev/certs/"
