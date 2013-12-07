#include <vector>
#include <iostream>
#include <algorithm>
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
    Pla();
    int train(const TrainElemAry&); ///< @return Number of updates happened
    void dumpModel(ostream&) const;
private:
    static int sign(double x) { return (x > 1e-15) ? 1 : -1; }
    bool isCorrect(const TrainElem&) const;
    int getWX(const TrainElem&) const;
    double w[DIMENSION+1];
};

Pla::Pla()
{
    for(int i = 0; i < DIMENSION; ++i)
        w[i] = 0.0;
}

int Pla::train(const TrainElemAry& in)
{
    int nUpdate = 0;
    int nError;
    do
    {
        nError = 0;
        for(TrainElemAry::const_iterator i = in.begin(); i != in.end(); ++i)
        {
            if(isCorrect(*i))
                continue;
            ++nError;
            ++nUpdate;
            w[0] += i->y;
            for(int j = 0; j < DIMENSION; ++j)
                w[j+1] += (i->y * i->x[j]);
        }
    } while(nError);
    return nUpdate;
}

void Pla::dumpModel(ostream& out) const
{
    out << w[0];
    for(int i = 1; i <= DIMENSION; ++i)
        out << " + " << w[i] << " * x[" << i << "]";
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

int main(void)
{
    double x[DIMENSION];
    int y;
    TrainElemAry trainData;
    while(cin >> x[0])
    {
        for(int i = 1; i < DIMENSION; ++i)
            cin >> x[i];
        cin >> y;
        trainData.push_back(TrainElem(x, y));
    }

    int totalUpdate = 0;
    for(int i = 0; i < 2000; ++i)
    {
        Pla learner;
        std::random_shuffle(trainData.begin(), trainData.end());
        int nUpdate = learner.train(trainData);
        cout << nUpdate << " updates required to get model:\n";
        learner.dumpModel(cout);
        cout << endl;
        totalUpdate += nUpdate;
    }
    printf("Average number of updates required = %lf\n", (double)totalUpdate / 2000.0);
    return 0;
}
