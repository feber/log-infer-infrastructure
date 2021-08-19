ARG ELK_VERSION

# https://www.docker.elastic.co/
FROM docker.elastic.co/logstash/logstash:${ELK_VERSION}

# Add your logstash plugins setup here
RUN logstash-plugin install logstash-filter-json
RUN logstash-plugin install logstash-filter-rest
RUN logstash-plugin install logstash-filter-http
