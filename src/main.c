#include <stdio.h>
#include <dynarray.h>


int main(){
	dynarray arr;
	int32_t src[] = {1, 2, 3, 4, 45, 1, 0};
	pml_err_t err = PML_OK;
	arr = dynarray_create(src, 7, TYPE_INT32, &err);
	arr.print(&arr);
	return 0;
}
