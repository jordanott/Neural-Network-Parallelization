#include <omp.h>
#include <iostream>

using namespace std;

main(int argc, char *argv[])  {

	int n = 10;
	int chunk = 10;

	float a[10], b[10], result;

	/* Some initializations */
	
	for (int i=0; i < n; i++) {
		a[i] = i;
		b[i] = i;
	}
	result = 0;
	#pragma omp parallel for	
	for (int i=0; i < n; i++)
	{
		#pragma omp critical
		{
			result  += a[i] * b[i];
		}
	 	
	 int id = omp_get_thread_num();
	cout << id << endl;
	}
	

	for (int j = 0; j < n; ++j)
	{
		cout << result << endl;
	}

}