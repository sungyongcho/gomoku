NAME    := gomoku
COMPOSE := docker compose -f docker-compose.yml

# If 'build' is among the goals, strip it out and set BUILD_FLAG
ifneq (,$(filter build,$(MAKECMDGOALS)))
  override MAKECMDGOALS := $(filter-out build,$(MAKECMDGOALS))
  BUILD_FLAG := --build
endif


all: $(NAME)

$(NAME):
	$(COMPOSE) up $(BUILD_FLAG) front back cpp_server

valgrind:
	$(COMPOSE) up $(BUILD_FLAG) front back cpp_server_valgrind

clean:
	$(COMPOSE) down

fclean: clean
	$(COMPOSE) rm -f front back cpp_server cpp_server_valgrind

re: fclean all

.PHONY: $(NAME) all valgrind clean fclean re
