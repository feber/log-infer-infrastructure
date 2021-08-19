echo "Setting default replica of 'logstash*' indexes to 0..."
curl -XPUT "localhost:9200/_template/all" -H "Content-Type: application/json" -d'
{
  "template": "logstash*",
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  }
}'
