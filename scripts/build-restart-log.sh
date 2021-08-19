docker-compose build $@ && \
  docker-compose restart $@ && \
  sleep 3 && \
  docker-compose logs $@
