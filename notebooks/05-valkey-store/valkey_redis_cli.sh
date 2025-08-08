ACCESS_TOKEN=$(gcloud auth print-access-token)
redis-cli --tls   --cacert server_ca.pem   -h 10.128.15.206   -p 6379   -a $ACCESS_TOKEN
# valkey-cli --tls   --cacert server_ca.pem   -h 10.128.15.206   -p 6379   -a $ACCESS_TOKEN
# any one of the above commands work, if you have valkey-cli installed use that
# redis-cli is relatively easy to install
# both function the same way so does not matter which one you use
