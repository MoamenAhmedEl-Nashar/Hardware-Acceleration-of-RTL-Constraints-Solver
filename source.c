


#include <stdio.h>
#include <math.h>
int f(int x)
{
	int temp = floor(x / (pow(x, 2) + 1));
	return temp + 1;
}
int g(int x)
{
	int temp = floor(-x / (pow(x, 2) + 1));
	return -temp ;
}
void model(int n_variables, int n_iterations)//add ranges , eqn
{
	for (int i = 0; i < n_iterations; i++)
	{
		int minimum_number = 0;
		int max_number = 10;
		int x = rand() % (max_number + 1 - minimum_number) + minimum_number;
		int y= rand() % (max_number + 1 - minimum_number) + minimum_number;
		int eqn = f(100 - pow(x, 2) - pow(y, 2));
		if (eqn == 1)
		{
			printf("%d , %d",x,y);
		}

	}
	printf("done");
}



