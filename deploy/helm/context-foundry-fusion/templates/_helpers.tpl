{{/*
Expand the name of the chart.
*/}}
{{- define "context-foundry-fusion.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name.
*/}}
{{- define "context-foundry-fusion.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version as used by the chart label.
*/}}
{{- define "context-foundry-fusion.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "context-foundry-fusion.labels" -}}
helm.sh/chart: {{ include "context-foundry-fusion.chart" . }}
{{ include "context-foundry-fusion.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "context-foundry-fusion.selectorLabels" -}}
app.kubernetes.io/name: {{ include "context-foundry-fusion.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ServiceAccount name to use.
*/}}
{{- define "context-foundry-fusion.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "context-foundry-fusion.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Name of the sensors ConfigMap (existing one wins).
*/}}
{{- define "context-foundry-fusion.configMapName" -}}
{{- if .Values.config.existingConfigMap }}
{{- .Values.config.existingConfigMap }}
{{- else }}
{{- printf "%s-sensors" (include "context-foundry-fusion.fullname" .) }}
{{- end }}
{{- end }}

{{/*
In-container path to the mounted --config JSON.
*/}}
{{- define "context-foundry-fusion.configPath" -}}
/etc/context-foundry/sensors.json
{{- end }}

{{/*
Directory the TAK CA bundle Secret is mounted at. Deliberately NOT under
/etc/context-foundry, which is already a configMap mount.
*/}}
{{- define "context-foundry-fusion.caDir" -}}
/etc/context-foundry-tak-ca
{{- end }}

{{/*
In-container path to the mounted TAK CA bundle (dir + the Secret's key).
*/}}
{{- define "context-foundry-fusion.caPath" -}}
{{- printf "%s/%s" (include "context-foundry-fusion.caDir" .) .Values.sinks.takWs.caSecret.key }}
{{- end }}

{{/*
Is a TAK CA bundle Secret configured?
*/}}
{{- define "context-foundry-fusion.caEnabled" -}}
{{- if .Values.sinks.takWs.caSecret.name }}true{{- end }}
{{- end }}

{{/*
Is a live UDP source enabled (i.e. does the pod listen on a socket)?
*/}}
{{- define "context-foundry-fusion.liveSource" -}}
{{- if or .Values.sources.sapient.enabled .Values.sources.cot.enabled }}true{{- end }}
{{- end }}

{{/*
Does a live UDP source require a Service? A host-level ingress path (hostPorts /
hostNetwork) already delivers the datagrams, so no Service is rendered for it
unless service.enabled asks for one explicitly.
*/}}
{{- define "context-foundry-fusion.needsService" -}}
{{- $hostIngress := or .Values.hostPorts.enabled .Values.hostNetwork.enabled }}
{{- if or .Values.service.enabled (and (include "context-foundry-fusion.liveSource" .) (not $hostIngress)) }}true{{- end }}
{{- end }}

{{/*
Pod dnsPolicy: explicit value wins, else ClusterFirstWithHostNet whenever
hostNetwork is on (without it the pod resolves against the node's resolv.conf
and the in-cluster TAK service name does not resolve).
*/}}
{{- define "context-foundry-fusion.dnsPolicy" -}}
{{- if .Values.dnsPolicy }}
{{- .Values.dnsPolicy }}
{{- else if .Values.hostNetwork.enabled }}
{{- "ClusterFirstWithHostNet" }}
{{- else }}
{{- "ClusterFirst" }}
{{- end }}
{{- end }}

{{/*
Is the OIDC client secret available (existing Secret or inline value)?
*/}}
{{- define "context-foundry-fusion.oidcSecretEnabled" -}}
{{- $s := .Values.sinks.takWs.auth.oidcClientSecret }}
{{- if or $s.secretName $s.value }}true{{- end }}
{{- end }}

{{/*
Is a static bearer token available (existing Secret or inline value)?
*/}}
{{- define "context-foundry-fusion.bearerEnabled" -}}
{{- $b := .Values.sinks.takWs.auth.bearerToken }}
{{- if or $b.secretName $b.value }}true{{- end }}
{{- end }}

{{/*
Resolved Secret name backing the OIDC client secret (existing wins, else chart-owned).
*/}}
{{- define "context-foundry-fusion.oidcSecretName" -}}
{{- $s := .Values.sinks.takWs.auth.oidcClientSecret }}
{{- default (include "context-foundry-fusion.fullname" .) $s.secretName }}
{{- end }}

{{/*
Resolved Secret name backing the bearer token (existing wins, else chart-owned).
*/}}
{{- define "context-foundry-fusion.bearerSecretName" -}}
{{- $b := .Values.sinks.takWs.auth.bearerToken }}
{{- default (include "context-foundry-fusion.fullname" .) $b.secretName }}
{{- end }}

{{/*
Does the chart need to render its own Secret (an inline value with no existing Secret)?
*/}}
{{- define "context-foundry-fusion.ownsSecret" -}}
{{- $s := .Values.sinks.takWs.auth.oidcClientSecret }}
{{- $b := .Values.sinks.takWs.auth.bearerToken }}
{{- if or (and $s.value (not $s.secretName)) (and $b.value (not $b.secretName)) }}true{{- end }}
{{- end }}

{{/*
One tracker bound as a string, empty when unset (null, or no tracker map at all)
so callers can treat "unset" and "leave the app's default" as the same thing.
Args: dict "values" .Values.tracker "key" <name>.
*/}}
{{- define "context-foundry-fusion.trackerBound" -}}
{{- $v := get (default dict .values) .key }}
{{- if not (kindIs "invalid" $v) }}{{ $v }}{{ end }}
{{- end }}

{{/*
Source/sink contract validation. Fails template rendering with a clear message
if there are zero sources or zero sinks configured.
*/}}
{{- define "context-foundry-fusion.validate" -}}
{{- $srcCount := 0 }}
{{- if .Values.sources.replayFile }}{{- $srcCount = add1 $srcCount }}{{- end }}
{{- if .Values.sources.sapient.enabled }}{{- $srcCount = add1 $srcCount }}{{- end }}
{{- if .Values.sources.cot.enabled }}{{- $srcCount = add1 $srcCount }}{{- end }}
{{- if eq $srcCount 0 }}
{{- fail "context-foundry-fusion: no SOURCE configured. Enable at least one of sources.replayFile, sources.sapient.enabled, sources.cot.enabled." }}
{{- end }}
{{- if and .Values.sources.replayFile (or .Values.sources.sapient.enabled .Values.sources.cot.enabled) }}
{{- fail "context-foundry-fusion: sources.replayFile cannot be combined with a live source (sources.sapient/sources.cot) -- a replay does not share a timeline with live sensors, and the app refuses the combination. Blank sources.replayFile to go live." }}
{{- end }}
{{- if not .Values.sinks.takWs.enabled }}
{{- fail "context-foundry-fusion: no SINK configured. Enable sinks.takWs.enabled (the WebTAK WebSocket sink)." }}
{{- end }}
{{- /* --- sink auth: the WS sink refuses to connect unauthenticated --- */}}
{{- if .Values.sinks.takWs.enabled }}
{{- $auth := .Values.sinks.takWs.auth }}
{{- if and $auth.keycloakTokenUrl (not $auth.oidcClientId) }}
{{- fail "context-foundry-fusion: sinks.takWs.auth.keycloakTokenUrl is set but sinks.takWs.auth.oidcClientId is empty -- the chart would render `--oidc-client-id ''`, which the WS sink treats as no client id at all and rejects at startup. Set sinks.takWs.auth.oidcClientId to the Keycloak client (per-group cf-<group>)." }}
{{- end }}
{{- if not (or (and $auth.keycloakTokenUrl $auth.oidcClientId) (include "context-foundry-fusion.bearerEnabled" .)) }}
{{- fail "context-foundry-fusion: the WebTAK-WS sink has no AUTH configured, and it cannot connect without one -- the container would exit at startup, so the chart refuses to render. Set EITHER the Keycloak client_credentials trio: sinks.takWs.auth.keycloakTokenUrl, sinks.takWs.auth.oidcClientId, sinks.takWs.auth.oidcClientSecret.secretName (or .value for dev) -- OR a static bearer token: sinks.takWs.auth.bearerToken.secretName (or .value)." }}
{{- end }}
{{- end }}
{{- /* --- tracker bounds: counts, and 0 would silently stop all tracking --- */}}
{{- range $k := list "maxLiveTracks" "maxTrackHistory" "maxEventDetections" }}
{{- $v := include "context-foundry-fusion.trackerBound" (dict "values" $.Values.tracker "key" $k) }}
{{- if and $v (le (int $v) 0) }}
{{- fail (printf "context-foundry-fusion: tracker.%s must be a positive count (got %v) -- 0 or less leaves the tracker with nothing to keep or nothing to associate, and the picture is silently empty. Leave it null to take the app's own default." $k $v) }}
{{- end }}
{{- end }}
{{- /* --- the coast horizon is seconds, not a count: compare as a float so a sub-second value survives --- */}}
{{- $coast := include "context-foundry-fusion.trackerBound" (dict "values" .Values.tracker "key" "maxCoastSeconds") }}
{{- if and $coast (le (float64 $coast) 0.0) }}
{{- fail (printf "context-foundry-fusion: tracker.maxCoastSeconds must be > 0 (got %v) -- a track that cannot coast at all is dropped as soon as the sweep that made it ends, so every sweep re-initiates the whole picture and nothing is fused across sensors. It has to exceed the sensors' revisit interval. Leave it null to take the app's own default." $coast) }}
{{- end }}
{{- /* --- sensor ingress --- */}}
{{- $live := include "context-foundry-fusion.liveSource" . }}
{{- if and .Values.hostNetwork.enabled .Values.hostPorts.enabled }}
{{- fail "context-foundry-fusion: hostNetwork.enabled and hostPorts.enabled are mutually exclusive -- with hostNetwork the container already binds the node's ports directly, so a hostPort mapping has nothing to map. Pick one." }}
{{- end }}
{{- if and .Values.service.enabled (not $live) }}
{{- fail "context-foundry-fusion: service.enabled is true but no live source is enabled -- the only Service ports are the sensor listeners (UDP 5000 sapient, UDP 6969 cot), so the Service would have no ports and the API server would reject it. The WS sink is outbound and needs no Service." }}
{{- end }}
{{- if and (or .Values.hostNetwork.enabled .Values.hostPorts.enabled) (not $live) }}
{{- fail "context-foundry-fusion: hostNetwork/hostPorts claim node ports for the sensor listeners, but no live source is enabled. Enable sources.sapient.enabled and/or sources.cot.enabled, or turn them off." }}
{{- end }}
{{- if eq .Values.service.type "ClusterIP" }}
{{- if or .Values.service.nodePorts.sapient .Values.service.nodePorts.cot }}
{{- fail "context-foundry-fusion: service.nodePorts is set but service.type is ClusterIP, which has no node ports -- the pinned port would be silently ignored and sensors would still have nowhere to send. Set service.type=NodePort (or use hostPorts/hostNetwork)." }}
{{- end }}
{{- end }}
{{- if and .Values.service.nodePorts.sapient (not .Values.sources.sapient.enabled) }}
{{- fail "context-foundry-fusion: service.nodePorts.sapient is set but sources.sapient.enabled is false -- no SAPIENT port is published, so the pin does nothing." }}
{{- end }}
{{- if and .Values.service.nodePorts.cot (not .Values.sources.cot.enabled) }}
{{- fail "context-foundry-fusion: service.nodePorts.cot is set but sources.cot.enabled is false -- no CoT port is published, so the pin does nothing." }}
{{- end }}
{{- if and .Values.sources.cot.enabled (ne .Values.service.type "ClusterIP") (eq .Values.service.externalTrafficPolicy "Cluster") }}
{{- fail "context-foundry-fusion: sources.cot.enabled with service.externalTrafficPolicy=Cluster silently corrupts the fused picture. kube-proxy masquerades the source address to the node IP, and the CoT source keys frame assembly on the sender's peer address (a CoT event's uid names the object observed, not the sender), so every external emitter collapses to one key, their reports merge into single events and the associator initiates duplicate tracks from them. Set service.externalTrafficPolicy=Local, or take the sensor traffic in via hostPorts/hostNetwork." }}
{{- end }}
{{- end }}

{{/*
================================================================================
ARGV BUILDER — emits the container args (argv) as a YAML list, in contract order:
  --config, then sources, then sinks, then extraArgs. Only enabled flags emit.
Secrets are referenced as $(VAR) — the literal is NEVER placed in args.
================================================================================
*/}}
{{- define "context-foundry-fusion.args" -}}
- --config
- {{ include "context-foundry-fusion.configPath" . | quote }}
{{- /* --- sources --- */}}
{{- if .Values.sources.replayFile }}
- --replay-file
- {{ .Values.sources.replayFile | quote }}
- --realtime-factor
- {{ .Values.sources.realtimeFactor | quote }}
{{- if .Values.sources.loop.enabled }}
- --loop
- --loop-delay
- {{ .Values.sources.loop.delaySeconds | quote }}
{{- end }}
{{- end }}
{{- if .Values.sources.sapient.enabled }}
- --enable-sapient
{{- end }}
{{- if .Values.sources.cot.enabled }}
- --enable-cot
{{- end }}
{{- /* kindIs, not truthiness: an explicit 0 must reach the app, not be dropped. */}}
{{- if and (include "context-foundry-fusion.liveSource" .) (not (kindIs "invalid" .Values.sources.frameWindowSeconds)) }}
- --frame-window-seconds
- {{ .Values.sources.frameWindowSeconds | quote }}
{{- end }}
{{- /* --- tracker bounds (both sources; null keeps the app's own default) --- */}}
{{- $maxLiveTracks := include "context-foundry-fusion.trackerBound" (dict "values" .Values.tracker "key" "maxLiveTracks") }}
{{- if $maxLiveTracks }}
- --max-live-tracks
- {{ $maxLiveTracks | quote }}
{{- end }}
{{- $maxTrackHistory := include "context-foundry-fusion.trackerBound" (dict "values" .Values.tracker "key" "maxTrackHistory") }}
{{- if $maxTrackHistory }}
- --max-track-history
- {{ $maxTrackHistory | quote }}
{{- end }}
{{- $maxCoastSeconds := include "context-foundry-fusion.trackerBound" (dict "values" .Values.tracker "key" "maxCoastSeconds") }}
{{- if $maxCoastSeconds }}
- --max-coast-seconds
- {{ $maxCoastSeconds | quote }}
{{- end }}
{{- $maxEventDetections := include "context-foundry-fusion.trackerBound" (dict "values" .Values.tracker "key" "maxEventDetections") }}
{{- if $maxEventDetections }}
- --max-event-detections
- {{ $maxEventDetections | quote }}
{{- end }}
{{- /* --- sink: WebTAK WebSocket --- */}}
{{- if .Values.sinks.takWs.enabled }}
- --tak-ws-host
- {{ .Values.sinks.takWs.host | quote }}
- --tak-ws-port
- {{ .Values.sinks.takWs.port | quote }}
{{- if include "context-foundry-fusion.caEnabled" . }}
- --tak-ws-ca
- {{ include "context-foundry-fusion.caPath" . | quote }}
{{- end }}
{{- if .Values.sinks.takWs.verifyTls }}
- --tak-ws-verify-tls
{{- end }}
{{- if .Values.sinks.takWs.auth.keycloakTokenUrl }}
- --keycloak-token-url
- {{ .Values.sinks.takWs.auth.keycloakTokenUrl | quote }}
- --oidc-client-id
- {{ .Values.sinks.takWs.auth.oidcClientId | quote }}
{{- if not (kindIs "invalid" .Values.sinks.takWs.auth.tokenTimeoutSeconds) }}
- --tak-ws-token-timeout
- {{ .Values.sinks.takWs.auth.tokenTimeoutSeconds | quote }}
{{- end }}
{{- if include "context-foundry-fusion.oidcSecretEnabled" . }}
- --oidc-client-secret
- $(OIDC_CLIENT_SECRET)
{{- end }}
{{- else if include "context-foundry-fusion.bearerEnabled" . }}
- --tak-bearer-token
- $(TAK_BEARER_TOKEN)
{{- end }}
{{- end }}
{{- /* --- cot framing --- */}}
- --cot-stale-seconds
- {{ .Values.cot.staleSeconds | quote }}
{{- /* --- extra --- */}}
{{- range .Values.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}

{{/*
================================================================================
ENV BUILDER — declares only the secret env vars that back $(VAR) references in
args. Values come from secretKeyRef, so the literal never lands in the pod spec.
================================================================================
*/}}
{{- define "context-foundry-fusion.env" -}}
{{- if and .Values.sinks.takWs.enabled .Values.sinks.takWs.auth.keycloakTokenUrl (include "context-foundry-fusion.oidcSecretEnabled" .) }}
- name: OIDC_CLIENT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "context-foundry-fusion.oidcSecretName" . }}
      key: {{ .Values.sinks.takWs.auth.oidcClientSecret.secretKey | quote }}
{{- end }}
{{- if and .Values.sinks.takWs.enabled (not .Values.sinks.takWs.auth.keycloakTokenUrl) (include "context-foundry-fusion.bearerEnabled" .) }}
- name: TAK_BEARER_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "context-foundry-fusion.bearerSecretName" . }}
      key: {{ .Values.sinks.takWs.auth.bearerToken.secretKey | quote }}
{{- end }}
{{- end }}

