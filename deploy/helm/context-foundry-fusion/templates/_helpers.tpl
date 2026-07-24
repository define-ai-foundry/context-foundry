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
Does a live UDP source require a Service?
*/}}
{{- define "context-foundry-fusion.needsService" -}}
{{- if or .Values.service.enabled .Values.sources.sapient.enabled .Values.sources.cot.enabled }}true{{- end }}
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
{{- if not .Values.sinks.takWs.enabled }}
{{- fail "context-foundry-fusion: no SINK configured. Enable sinks.takWs.enabled (the WebTAK WebSocket sink)." }}
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
{{- /* --- sink: WebTAK WebSocket --- */}}
{{- if .Values.sinks.takWs.enabled }}
- --tak-ws-host
- {{ .Values.sinks.takWs.host | quote }}
- --tak-ws-port
- {{ .Values.sinks.takWs.port | quote }}
{{- if .Values.sinks.takWs.verifyTls }}
- --tak-ws-verify-tls
{{- end }}
{{- if .Values.sinks.takWs.auth.keycloakTokenUrl }}
- --keycloak-token-url
- {{ .Values.sinks.takWs.auth.keycloakTokenUrl | quote }}
- --oidc-client-id
- {{ .Values.sinks.takWs.auth.oidcClientId | quote }}
{{- if include "context-foundry-fusion.oidcSecretEnabled" . }}
- --oidc-client-secret
- $(OIDC_CLIENT_SECRET)
{{- end }}
{{- else if include "context-foundry-fusion.bearerEnabled" . }}
- --tak-bearer-token
- $(TAK_BEARER_TOKEN)
{{- end }}
{{- end }}
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

