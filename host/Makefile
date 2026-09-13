CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Wno-unused-parameter -I../src
LDFLAGS ?= -lm
SRC = ../src/needle.c ../src/needle_cq.c ../src/needle_tok.c main.c
needle_host: $(SRC) ../src/needle.h ../src/needle_cq.h ../src/nsp.h
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
clean:
	rm -f needle_host
