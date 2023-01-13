CFLAGS= -O3 -g3 -c -Wall
CC= gcc
SRC = src
LIB = lib
OBJ = obj
BIN = bin
OBJECTS = $(OBJ)/utilities.o  $(OBJ)/lstm.o
HEADERS = $(SRC)/utilities.h  $(SRC)/lstm.h  $(SRC)/std_conf.h

all : app test

$(OBJ)/utilities.o: $(SRC)/utilities.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/utilities.c  -o  $(OBJ)/utilities.o -lm -lpthread

$(OBJ)/lstm.o: $(SRC)/lstm.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/lstm.c  -o  $(OBJ)/lstm.o -lm -lpthread

$(OBJ)/app.o: $(SRC)/app.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/app.c  -o $(OBJ)/app.o -lm -lpthread

$(OBJ)/test.o: $(SRC)/test.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/test.c -o $(OBJ)/test.o -lm -lpthread

app: ${OBJECTS} $(OBJ)/app.o $(HEADERS)
	$(CC) -o $(BIN)/app.exe ${OBJECTS} $(OBJ)/app.o -lm -lpthread

test: ${OBJECTS} $(OBJ)/test.o $(HEADERS)
	$(CC) -o $(BIN)/test.exe ${OBJECTS} $(OBJ)/test.o -lm -lpthread

clean: 
	rm -rf $(OBJ)/*.o
	rm -rf $(BIN)/*.exe

empty: 
	del /F /Q $(OBJ)\*.o
	del /F /Q $(BIN)\*.exe

run-test: test
	./$(BIN)/test.exe

run-app: app
	./$(BIN)/app.exe
