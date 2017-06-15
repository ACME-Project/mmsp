CPP = g++
PCPP = mpic++
CPPOPTIONS = -O3 -I $(MMSP_PATH)/include -std=gnu++11
PCPPOPTIONS = $(CPPOPTIONS) -include mpi.h
LIBS = -lm -lz

#Program dependencies below here.
all : mpf-test
.PHONY : all

mpf-test : mpf-test.cpp
	$(CPP) $(CPPOPTIONS) $< -o $@ $(LIBS)

parallel : mpf-test.cpp
	$(PCPP) $(PCPPOPTIONS) $< -o $@ $(LIBS)

clean :
	rm *.o *~ mpf-test parallel *.vti *.dat *.tsv
