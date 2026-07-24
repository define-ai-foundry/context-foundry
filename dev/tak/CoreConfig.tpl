<?xml version="1.0" encoding="UTF-8"?>
<!-- Local-dev CoreConfig for the containerized TAK Server, rendered by gomplate
     at container start. Provides the 8089 mutual-TLS CoT input, the 8446 certless
     connector (WebTAK login + the /takproto/1 streaming WebSocket), and file-based
     auth. Not for production use. -->
<Configuration xmlns="http://bbn.com/marti/xml/config"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="/opt/tak/CoreConfig.xsd">
    <network multicastTTL="5">
        <!-- Direct mutual-TLS CoT input (port 8089): requires a CA-signed client
             cert (auth="x509", authRequired="true"). Group assignment on this input
             is server-side (see docs/tak-keycloak-group-isolation.md). -->
        <input _name="stdssl" protocol="tls" port="8089" coreVersion="2" auth="x509" authRequired="true"/>

        <connector port="8443" _name="https" enableWebtak="{{getenv "WEBTAK_ENABLE" "true"}}" enableNonAdminUI="false" />
        <!-- Certless connector (clientAuth="NONE", no OIDC): serves WebTAK username/
             password login AND the /takproto/1 streaming WebSocket that TakWsSink
             uses. CoT pushed over it is tagged with the caller's TAK groups. -->
        <connector port="8446" _name="cert_https" clientAuth="NONE" enableWebtak="true" enableAdminUI="true" enableNonAdminUI="true" />
    </network>

    <auth x509groups="true" x509addAnonymous="false">
        <File location="/opt/tak/data/UserAuthenticationFile.xml"/>
    </auth>

    <submission ignoreStaleMessages="false" validateXml="false"/>

    <subscription reloadPersistent="false">
    </subscription>

    <repository enable="true" numDbConnections="50" primaryKeyBatchSize="500" insertionBatchSize="500">
      <connection url="jdbc:postgresql://{{getenv "POSTGRES_ADDRESS" "tak-database"}}:5432/{{getenv "POSTGRES_DB" "cot"}}" username="{{.Env.POSTGRES_USER}}" password="{{.Env.POSTGRES_PASSWORD}}" />
    </repository>

    <repeater enable="true" periodMillis="3000" staleDelayMillis="15000">
        <repeatableType initiate-test="/event/detail/emergency[@type='911 Alert']" cancel-test="/event/detail/emergency[@cancel='true']" _name="911"/>
        <repeatableType initiate-test="/event/detail/emergency[@type='Ring The Bell']" cancel-test="/event/detail/emergency[@cancel='true']" _name="RingTheBell"/>
        <repeatableType initiate-test="/event/detail/emergency[@type='Geo-fence Breached']" cancel-test="/event/detail/emergency[@cancel='true']" _name="GeoFenceBreach"/>
        <repeatableType initiate-test="/event/detail/emergency[@type='Troops In Contact']" cancel-test="/event/detail/emergency[@cancel='true']" _name="TroopsInContact"/>
    </repeater>

    <dissemination smartRetry="false" />

    <filter>
        <flowtag enable="false" text=""/>
        <streamingbroker enable="true"/>
        <scrubber enable="false" action="overwrite"/>
    </filter>

    <buffer>
        <queue enableStoreForwardChat="true">
            <priority/>
        </queue>
        <latestSA enable="true"/>
    </buffer>

    <security>
        <tls context="TLSv1.2"
            keymanager="SunX509"
            keystore="JKS" keystoreFile="/opt/tak/data/certs/files/takserver.jks" keystorePass="{{.Env.TAKSERVER_CERT_PASS}}"
            truststore="JKS" truststoreFile="/opt/tak/data/certs/files/truststore-root.jks" truststorePass="{{.Env.CA_PASS}}"
            enableOCSP="{{getenv "TAK_OCSP_ENABLE" "false"}}"
            />
    </security>

    <logging
        auditLoggingEnabled="true"
        httpAccessEnabled="true"
        jsonFormatEnabled="true"
    />

</Configuration>
