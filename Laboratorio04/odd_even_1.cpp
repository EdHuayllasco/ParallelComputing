#include <iostream>
#include <vector>
#include <chrono>

const int RMAX = 10000;

void Usage(char *prog_name);
void Get_args(int argc, char *argv[], int *n_p, char *g_i_p);
void Generate_list(std::vector<int> &a, int n);
void Print_list(const std::vector<int> &a, const char *title);
void Read_list(std::vector<int> &a, int n);
void Odd_even_sort_parallel(std::vector<int> &a, int thread_count);

int main(int argc, char *argv[])
{
    int n;
    char g_i;

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <n> <g|i>\n";
        return 1;
    }

    n = std::atoi(argv[1]);
    g_i = argv[2][0];

    if (n <= 0 || (g_i != 'g' && g_i != 'i'))
    {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    std::vector<int> a(n);

    if (g_i == 'g')
    {
        Generate_list(a, n);
        // Print_list(a, "Before sort");
    }
    else
    {
        Read_list(a, n);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    Odd_even_sort_parallel(a, 12); // Adjust the thread count as needed

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Print_list(a, "After sort");

    std::cout << "Time taken: " << elapsed_time.count() << " seconds\n";

    return 0;
}

void Usage(char *prog_name)
{
    std::cerr << "usage:   " << prog_name << " <n> <g|i>\n";
    std::cerr << "   n:   number of elements in list\n";
    std::cerr << "  'g':  generate list using a random number generator\n";
    std::cerr << "  'i':  user input list\n";
}

void Generate_list(std::vector<int> &a, int n)
{
    srand(0);
    for (int i = 0; i < n; ++i)
    {
        a[i] = rand() % RMAX;
    }
}

void Print_list(const std::vector<int> &a, const char *title)
{
    std::cout << title << ":\n";
    for (int val : a)
    {
        std::cout << val << " ";
    }
    std::cout << "\n\n";
}

void Read_list(std::vector<int> &a, int n)
{
    std::cout << "Please enter the elements of the list\n";
    for (int i = 0; i < n; ++i)
    {
        std::cin >> a[i];
    }
}

void Odd_even_sort_parallel(std::vector<int> &a, int thread_count)
{
    int phase, i, tmp;

#pragma omp parallel num_threads(thread_count) default(none) shared(a, phase) private(i, tmp)
    for (phase = 0; phase < a.size(); phase++)
    {
        if (phase % 2 == 0)
        {
#pragma omp for
            for (i = 1; i < a.size(); i += 2)
            {
                if (a[i - 1] > a[i])
                {
                    tmp = a[i - 1];
                    a[i - 1] = a[i];
                    a[i] = tmp;
                }
            }
        }
        else
        {
#pragma omp for
            for (i = 1; i < a.size() - 1; i += 2)
            {
                if (a[i] > a[i + 1])
                {
                    tmp = a[i + 1];
                    a[i + 1] = a[i];
                    a[i] = tmp;
                }
            }
        }
    }
}
