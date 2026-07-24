#!/usr/bin/env bash
# All-in-one local launch for the stock pvarki/tak-server image. start-tak.sh
# renders CoreConfig per-role in separate containers and re-links certs/logs into
# the data volume on every start; configureInDocker.sh runs every role in one
# container but does neither. We do both here, then hand off to configureInDocker.
set -e

/opt/scripts/firstrun.sh

TR=/opt/tak
# Re-link certs/ and logs/ into the data volume on every start. firstrun.sh does
# this too, but only on first run; on a reused volume (fresh container) it early-
# exits, leaving the image's plain certs/ dir to shadow the volume and break
# relative-path lookups such as the federation truststore (certs/files/*.jks).
mkdir -p "${TR}/data/certs" "${TR}/data/logs"
if [[ ! -L "${TR}/certs" ]]; then
  rm -rf "${TR}/certs.orig"; mv "${TR}/certs" "${TR}/certs.orig"
  ln -s "${TR}/data/certs/" "${TR}/certs"
fi
if [[ ! -L "${TR}/logs" ]]; then
  rm -rf "${TR}/logs.orig"; mv "${TR}/logs" "${TR}/logs.orig"
  ln -s "${TR}/data/logs/" "${TR}/logs"
fi

cd /opt/tak
gomplate -f /opt/templates/CoreConfig.tpl -o /opt/tak/data/CoreConfig.xml
ln -sf /opt/tak/data/CoreConfig.xml /opt/tak/CoreConfig.xml
gomplate -f /opt/templates/TAKIgniteConfig.tpl -o /opt/tak/data/TAKIgniteConfig.xml
ln -sf /opt/tak/data/TAKIgniteConfig.xml /opt/tak/TAKIgniteConfig.xml
cp /opt/templates/logback-stdout.xml /opt/tak/

exec /opt/tak/configureInDocker.sh run
