# Name of this mp
MP_NAME = template

# define LIBWB_PATH
LIBWB_PATH = /home/haoze/libwb
# SO file path
LIBWB_SO_PATH = $(LIBWB_PATH)/lib/libwb.so

MP_OBJ = $(MP_NAME).o

$(MP_NAME): $(MP_OBJ)
	nvcc -std=c++11 -g -G -o $(MP_NAME) $(MP_OBJ) $(LIBWB_SO_PATH)
$(MP_OBJ): template.cu
	nvcc -std=c++11 -g -G -I $(LIBWB_PATH) -rdc=true -c template.cu -o $(MP_OBJ)
# run the program
PHONY: run
run:
	bash run_datasets

# clean
PHONY: clean
clean:
	rm -f $(MP_NAME) *.o