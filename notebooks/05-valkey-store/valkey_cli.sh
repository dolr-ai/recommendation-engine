ACCESS_TOKEN=$(gcloud auth print-access-token)
valkey-cli --tls   --cacert server_ca.pem   -h 10.128.15.206   -p 6379   -a $ACCESS_TOKEN