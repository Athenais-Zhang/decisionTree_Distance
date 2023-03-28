// SequenceEnsemble.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <string>
#include <vector>
#include "SequenceReader.h"
#include <set>
#include <future>
#include <thread>
using namespace std;
float longest_common_substring_similarity(vector<string> a, vector<string> b){
	int length = 0;
	int** dp = new int*[a.size() + 1];
	for (int i = 0; i < a.size() + 1; i++)
	{
		dp[i] = new int[b.size() + 1];
	}
	for (int i = 0; i < a.size() + 1; i++)
	{
		for (int j = 0; j < b.size() + 1; j++)
		{
			dp[i][j] = 0;
		}
	}
	for (int i = 1; i < a.size() + 1; i++)
	{
		for (int j = 1; j < b.size() + 1; j++)
		{
			if (a[i - 1] == b[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
				length = max(length, dp[i][j]+1);
			}
			else
				dp[i][j] = 0;
		}
	}
	// delete dp
	for (int i = 0; i < a.size() + 1; i++)
	{
		delete[] dp[i];
	}
	delete[] dp;

	// float length2 = 0;
	// length2 = (float)length / (float)max(a.size(), b.size());
	return (float)length;
}

float subsequence_similarity(vector<string> a, vector<string> b){
// if b is the subsequenece of a, return 1, else return 0
	int i = 0;
	int j = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] == b[j])
		{
			i++;
			j++;
		}
		else
			i++;
	}
	if (j == b.size()){
	// pring a, b , 1
	// string total = "";
	// for (int i = 0; i < a.size(); i++)
	// {
	// 	total += a[i];
	// 	total += " ";
	// }
	// total += "----- ";
	// for (int i = 0; i < b.size(); i++)
	// {
	// 	total += b[i];
	// 	total += " ";
	// }
	// total += "1";
	// cout << total << endl;

		return 1;}
	else
		return 0;
}

	

class PairwiseSimilarity
{
public:
	string similarity_measure;
	vector<vector<string> > sequence;
	set<vector<string> > itemset;
	float (*similarity_function)(vector<string>, vector<string>);

	set<vector<string> > generate_features();
    PairwiseSimilarity(string filename,string similarity_measure);
    float **Cal_PairwiseSimilarity();

	void data_process(vector<vector<string> > tempsequence){
		// generate sequence, based on split " "
		for (int i = 0; i < tempsequence.size(); i++)
		{
			vector<string> onesequence;
			string emptyItem = "";
			for (int j = 0; j < tempsequence[i].size(); j++)
			{
				if (tempsequence[i][j] == " "){
					// add item to itemset
					onesequence.push_back(emptyItem);
					emptyItem = "";

				}
				else
					emptyItem += tempsequence[i][j];
			}
			// cout one sequence
			// string total = "";
			// for (int i = 0; i < onesequence.size(); i++)
			// {
			// 	total += onesequence[i];
			// 	total += " ";
			// }
			// cout << total << endl;
			this->sequence.push_back(onesequence);
		}
	}

private:
};

PairwiseSimilarity::PairwiseSimilarity(string filename, string similarity_measure)
{

    if (similarity_measure == "lcs") {
		cout << "similarity_measure is lcs" << endl;
        similarity_function = longest_common_substring_similarity;
    }
	else {
        cout << "similarity_measure is not supported" << endl;
    }
	// randomly
	// read sequence
	SequenceReader sequenceReader;
	pair<vector<string>, vector<vector<string> > > sequencePair = sequenceReader.readSequence(filename);
	vector<string> label = sequencePair.first;

	vector<vector<string> > tempsequence = sequencePair.second;
	// vector<vector<string>> sequence;
	this->data_process(tempsequence);
	float** similarity_matrix = Cal_PairwiseSimilarity();
	// save similarity matrix
	ofstream outfile;
	outfile.open("similarity_matrix.txt");
	for (int i = 0; i < this->sequence.size(); i++)
	{
		for (int j = 0; j < this->sequence.size(); j++)
		{
			outfile << similarity_matrix[i][j] << " ";
		}
		outfile << endl;
	}
	outfile.close();

}



// float** PairwiseSimilarity::Cal_PairwiseSimilarity()
// {
// 	// calculate the similarity matrix based on the feature vector: N*N matrix
// 	// start time 
// 	clock_t start, finish;
// 	double totaltime;
// 	start = clock();

// 	float** similarity_matrix = new float*[this->sequence.size()];
// 	for (int i = 0; i < this->sequence.size(); i++)
// 	{
// 		similarity_matrix[i] = new float[this->sequence.size()];
// 	}
// 	// version 1: single thread
// 	for (int i = 0; i < this->sequence.size(); i++)
// 	{
// 		for (int j = 0; j < this->sequence.size(); j++)
// 		{
// 			similarity_matrix[i][j] = similarity_function(this->sequence[i], this->sequence[j]);
// 			similarity_matrix[j][i] = similarity_matrix[i][j];
// 		}
// 	}
// 	// end time
// 	finish = clock();
// 	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
// 	cout << "single thread time: " << totaltime << "s" << endl;
// 	return similarity_matrix;
// }

float** PairwiseSimilarity::Cal_PairwiseSimilarity()
{
		// start time 
	clock_t start, finish;
	double totaltime;
	start = clock();
    // calculate the similarity matrix based on the feature vector: N*N matrix
    float** similarity_matrix = new float*[this->sequence.size()];
    for (int i = 0; i < this->sequence.size(); i++)
    {
        similarity_matrix[i] = new float[this->sequence.size()];
    }

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void> > futures(num_threads);

    const int block_size = this->sequence.size() / num_threads;

    // dispatch tasks to worker threads
    for (int t = 0; t < num_threads; t++)
    {
        int start = t * block_size;
        int end = (t + 1) * block_size;
        if (t == num_threads - 1)
        {
            end = this->sequence.size();
        }

        futures[t] = std::async(std::launch::async, [&, start, end] {
            for (int i = start; i < end; i++)
            {
                for (int j = 0; j <= i; j++) // only calculate upper triangular matrix
                {
                    similarity_matrix[i][j] = similarity_function(this->sequence[i], this->sequence[j]);
                    similarity_matrix[j][i] = similarity_matrix[i][j];
                }
            }
        });
    }

    // wait for all worker threads to finish
    for (auto& f : futures)
    {
        f.wait();
    }
	// end time
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "multi thread time: " << totaltime << "s" << endl;
    return similarity_matrix;
}

int main( int argc, char** argv )
{

	// string filename = "../dataset/gene.txt";
	// string similarity = "lcs";
	// 写成这样
	string filename = argv[1];
	filename = "dataset/" + filename + ".txt";
	// int iter_num = atoi(argv[2]);
	// int max_feature_num = atoi(argv[3]);
	// int max_feature_len = atoi(argv[4]);
	string similarity = argv[2];
	// SequenceEnsembleCluster sequenceEnsembleCluster(filename, iter_num, max_feature_num, max_feature_len, similarity);
	// 然后在python 里面用这个函数 subprocess.call('./pairdistacne.out ' + "gene" + ' ' + "lcs", shell=True)
	// 上面的 你改一改就行了
	PairwiseSimilarity PairwiseSimilarity(filename, similarity);
	return 0;
}




