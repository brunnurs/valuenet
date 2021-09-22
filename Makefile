.PHONY: up
up:
	docker-compose up

.PHONY: stop
stop:
	docker-compose stop

.PHONY: clean
clean:
	docker-compose down

.PHONY: jumpin-postgres
jumpin-postgres:
	docker-compose exec postgres /bin/bash

.PHONY: import-data
import-data:
	docker-compose exec postgres sh -c 'psql --dbname=hack_zurich --username=hack_zurich --password < /var/dumps/2021-09-20-hack_zurich.sql'
