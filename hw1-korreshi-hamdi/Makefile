COMP = mpicc
TARGET = template
SRC = template.c
RUN = mpirun

compile:
        $(COMP) $(SRC) -o $(TARGET)
run:
        $(RUN) -np 8 --hostfile host_file ./$(TARGET)
clean:
        rm template