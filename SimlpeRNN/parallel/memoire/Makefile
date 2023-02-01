CFLAGS= -O3 -g3 -c -Wall
CC= gcc
SRC = src
LIB = lib
OBJ = obj
BIN = bin
OBJECTS = $(OBJ)/utils.o $(OBJ)/simplernn.o
HEADERS = $(SRC)/utils.h  $(SRC)/simplernn.h

all : app test

$(OBJ)/utils.o: $(SRC)/utils.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/utils.c  -o  $(OBJ)/utils.o -lm -lpthread

$(OBJ)/simplernn.o: $(SRC)/simplernn.c $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/simplernn.c  -o  $(OBJ)/simplernn.o -lm -lpthread

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
