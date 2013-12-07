#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
using std::vector;
using std::cin;
using std::cout;
using std::endl;
using std::ostream;

static const int DIMENSION = 4;

struct TrainElem
{
    TrainElem(double p_x[DIMENSION], int p_y)
        : y(p_y)
    {
        for(int i = 0; i < DIMENSION; ++i)
            x[i] = p_x[i];
    }
    double x[DIMENSION];
    int y;
};

typedef vector<TrainElem> TrainElemAry;

class Pla
{
public:
    Pla(double p_eta = 1.0);
    int train(const TrainElemAry&, int updateLimit); ///< @return Number of updates happened
    void dumpModel(ostream&) const;
    int countError(const TrainElemAry&) const;
private:
    static int sign(double x) { return (x > 1e-15) ? 1 : -1; }
    bool isCorrect(const TrainElem&) const;
    int getWX(const TrainElem&) const;
    double w[DIMENSION+1];
    const double eta;
};

Pla::Pla(double p_eta) : eta(p_eta)
{
    for(int i = 0; i < DIMENSION; ++i)
        w[i] = 0.0;
}

int Pla::train(const TrainElemAry& in, int updateLimit)
{
    int nUpdate = 0;
    int nError;
    do
    {
        nError = 0;
        double backup_w[DIMENSION+1];
        for(int j = 0; j <= DIMENSION; ++j)
            backup_w[j] = w[j];
        int prevErr = countError(in);
        for(TrainElemAry::const_iterator i = in.begin(); i != in.end(); ++i)
        {
            if(isCorrect(*i))
                continue;

            w[0] += eta * i->y;
            for(int j = 0; j < DIMENSION; ++j)
                w[j+1] += (eta * i->y * i->x[j]);

            ++nError;
        }

        int curErr = countError(in);
        if(curErr >= prevErr) // worse, roll back
        {
            for(int j = 0; j <= DIMENSION; ++j)
                w[j] = backup_w[j];
        }
        ++nUpdate;
    } while(nError && nUpdate < updateLimit);
    return nUpdate;
}

void Pla::dumpModel(ostream& out) const
{
    out << w[0];
    for(int i = 1; i <= DIMENSION; ++i)
        out << " + " << w[i] << " * x[" << i << "]";
}

int Pla::countError(const TrainElemAry& in) const
{
    int ans = 0;
    for(TrainElemAry::const_iterator i = in.begin(); i != in.end(); ++i)
    {
        if(!isCorrect(*i))
            ++ans;
    }
    return ans;
}

bool Pla::isCorrect(const TrainElem& in) const
{
    return getWX(in) == in.y;
}

int Pla::getWX(const TrainElem& in) const
{
    double val = w[0];
    for(int i = 0; i < DIMENSION; ++i)
        val += (w[i+1] * in.x[i]);
    return sign(val);
}

int main(int argc, char *argv[])
{
    double x[DIMENSION];
    int y;
    TrainElemAry trainData;
    double eta = 1.0;
    if (argc > 1)
    {
        std::istringstream strm(argv[1]);
        strm >> eta;
    }
    while(cin >> x[0])
    {
        for(int i = 1; i < DIMENSION; ++i)
            cin >> x[i];
        cin >> y;
        trainData.push_back(TrainElem(x, y));
    }

    TrainElemAry verifyData;
    if (argc > 2)
    {
        std::ifstream verifyFile(argv[2]);
        while(verifyFile >> x[0])
        {
            for(int i = 1; i < DIMENSION; ++i)
                verifyFile >> x[i];
            verifyFile >> y;
            verifyData.push_back(TrainElem(x, y));
        }
    }

    int totalError = 0;
    for(int i = 0; i < 2000; ++i)
    {
        Pla learner(eta);
        std::random_shuffle(trainData.begin(), trainData.end());
        int nUpdate = learner.train(trainData, 100);
        cout << nUpdate << " updates required to get model:\n";
        learner.dumpModel(cout);
        cout << endl;
        totalError += learner.countError(verifyData);
    }
    printf("Average error rate = %lf\n", (double)totalError / 2000.0 / verifyData.size());
    return 0;
}
