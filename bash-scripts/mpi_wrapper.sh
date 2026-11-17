#!/bin/bash
IFACE=$(ip -4 route get 192.168.108.1 |
 awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1);
exit}}')
if [[ -z "$IFACE" || "$IFACE" == "lo" ]]; then
 echo "ERROR: No valid Ethernet interface found on $(hostname)"
>&2
 exit 1
fi
export UCX_TLS=tcp,self,sm
export UCX_NET_DEVICES="$IFACE"
echo "$(hostname): UCX_NET_DEVICES=$UCX_NET_DEVICES"
exec "$@"