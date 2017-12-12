#include "TriQuadrature.hpp"
#include "LineQuadrature.hpp"

#include <vector>
#include <cassert>
#include <cmath>

namespace poly_fem
{
    namespace
    {
        void get_weight_and_points(const int order, Eigen::MatrixXd &points, Eigen::MatrixXd &weights)
        {
            switch(order)
            {
                case 0: {
                    points.resize(1, 2);
                    weights.resize(1, 1);

                    points << 0.333333333333333333333333333333333, 0.333333333333333333333333333333333;
                    weights << 1;

                    break;
                }

                case 3: {
                    points.resize(6, 2);
                    weights.resize(6, 1);

                    points <<
                    0.5, 0.5,
                    0.5, 0.0,
                    0.0, 0.5,
                    0.16666666666666666666666666666667, 0.16666666666666666666666666666667,
                    0.16666666666666666666666666666667, 0.66666666666666666666666666666667,
                    0.66666666666666666666666666666667, 0.16666666666666666666666666666667;

                    weights <<
                    0.033333333333333333333333333333333,
                    0.033333333333333333333333333333333,
                    0.033333333333333333333333333333333,
                    0.3,
                    0.3,
                    0.3;

                    break;
                }

                case 4: {
                    points.resize(6, 2);
                    weights.resize(6, 1);

                    points <<
                    0.091576213509770743459571463402202, 0.091576213509770743459571463402202,
                    0.091576213509770743459571463402202, 0.81684757298045851308085707319560,
                    0.81684757298045851308085707319560, 0.091576213509770743459571463402202,
                    0.44594849091596488631832925388305, 0.44594849091596488631832925388305,
                    0.44594849091596488631832925388305, 0.10810301816807022736334149223390,
                    0.10810301816807022736334149223390, 0.44594849091596488631832925388305;

                    weights <<
                    0.10995174365532186763832632490021,
                    0.10995174365532186763832632490021,
                    0.10995174365532186763832632490021,
                    0.22338158967801146569500700843312,
                    0.22338158967801146569500700843312,
                    0.22338158967801146569500700843312;

                    break;
                }

                case 6: {
                    points.resize(11, 2);
                    weights.resize(11, 1);

                    points <<
                    0.063089014491502228340331602870819, 0.063089014491502228340331602870819,
                    0.063089014491502228340331602870819, 0.87382197101699554331933679425836,
                    0.87382197101699554331933679425836, 0.063089014491502228340331602870819,
                    0.24928674517091042129163855310702, 0.24928674517091042129163855310702,
                    0.24928674517091042129163855310702, 0.50142650965817915741672289378596,
                    0.50142650965817915741672289378596, 0.24928674517091042129163855310702,
                    0.053145049844816947353249671631398, 0.31035245103378440541660773395655,
                    0.053145049844816947353249671631398, 0.63650249912139864723014259441205,
                    0.31035245103378440541660773395655, 0.053145049844816947353249671631398,
                    0.31035245103378440541660773395655, 0.63650249912139864723014259441205,
                    0.63650249912139864723014259441205, 0.053145049844816947353249671631398,
                    0.63650249912139864723014259441205, 0.31035245103378440541660773395655;

                    weights <<
                    0.050844906370206816920936809106869,
                    0.050844906370206816920936809106869,
                    0.050844906370206816920936809106869,
                    0.11678627572637936602528961138558,
                    0.11678627572637936602528961138558,
                    0.11678627572637936602528961138558,
                    0.082851075618373575193553456420442,
                    0.082851075618373575193553456420442,
                    0.082851075618373575193553456420442,
                    0.082851075618373575193553456420442,
                    0.082851075618373575193553456420442,
                    0.082851075618373575193553456420442;

                    break;
                }

                case 7: {
                    points.resize(12, 2);
                    weights.resize(12, 1);

                    points <<
                    0.0623822650944021181736830009963499, 0.0675178670739160854425571310508685,
                    0.0675178670739160854425571310508685, 0.870099867831681796383759867952782,
                    0.870099867831681796383759867952782,  0.0623822650944021181736830009963499,
                    0.0552254566569266117374791902756449, 0.321502493851981822666307849199202,
                    0.321502493851981822666307849199202,  0.623272049491091565596212960525153,
                    0.623272049491091565596212960525153,  0.0552254566569266117374791902756449,
                    0.0343243029450971464696306424839376, 0.660949196186735657611980310197799,
                    0.660949196186735657611980310197799,  0.304726500868167195918389047318263,
                    0.304726500868167195918389047318263,  0.0343243029450971464696306424839376,
                    0.515842334353591779257463386826430,  0.277716166976391782569581871393723,
                    0.277716166976391782569581871393723,  0.20644149867001643817295474177985,
                    0.20644149867001643817295474177985,   0.515842334353591779257463386826430;

                    weights <<
                    0.053034056314872502857508360921478,
                    0.053034056314872502857508360921478,
                    0.053034056314872502857508360921478,
                    0.087762817428892110073539806278575,
                    0.087762817428892110073539806278575,
                    0.087762817428892110073539806278575,
                    0.057550085569963171476890993800437,
                    0.057550085569963171476890993800437,
                    0.057550085569963171476890993800437,
                    0.13498637401960554892539417233284,
                    0.13498637401960554892539417233284,
                    0.13498637401960554892539417233284;

                    break;
                }

                case 8: {
                    points.resize(16, 2);
                    weights.resize(16, 1);

                    points <<
                    0.33333333333333333333333333333333,  0.33333333333333333333333333333333,
                    0.17056930775176020662229350149146,  0.17056930775176020662229350149146,
                    0.17056930775176020662229350149146,  0.65886138449647958675541299701707,
                    0.65886138449647958675541299701707,  0.17056930775176020662229350149146,
                    0.050547228317030975458423550596599, 0.050547228317030975458423550596599,
                    0.050547228317030975458423550596599, 0.89890554336593804908315289880680,
                    0.89890554336593804908315289880680,  0.050547228317030975458423550596599,
                    0.45929258829272315602881551449417,  0.45929258829272315602881551449417,
                    0.45929258829272315602881551449417,  0.08141482341455368794236897101166,
                    0.08141482341455368794236897101166,  0.45929258829272315602881551449417,
                    0.72849239295540428124100037917606,  0.26311282963463811342178578628464,
                    0.72849239295540428124100037917606,  0.00839477740995760533721383453930,
                    0.26311282963463811342178578628464,  0.72849239295540428124100037917606,
                    0.26311282963463811342178578628464,  0.00839477740995760533721383453930,
                    0.00839477740995760533721383453930,  0.72849239295540428124100037917606,
                    0.00839477740995760533721383453930,  0.26311282963463811342178578628464;

                    weights <<
                    0.14431560767778716825109111048906,
                    0.10321737053471825028179155029213,
                    0.10321737053471825028179155029213,
                    0.10321737053471825028179155029213,
                    0.032458497623198080310925928341780,
                    0.032458497623198080310925928341780,
                    0.032458497623198080310925928341780,
                    0.095091634267284624793896104388584,
                    0.095091634267284624793896104388584,
                    0.095091634267284624793896104388584,
                    0.027230314174434994264844690073909,
                    0.027230314174434994264844690073909,
                    0.027230314174434994264844690073909,
                    0.027230314174434994264844690073909,
                    0.027230314174434994264844690073909,
                    0.027230314174434994264844690073909;

                    break;
                }

                case 9: {
                    points.resize(19, 2);
                    weights.resize(19, 1);

                    points <<
                    0.333333333333333333333333333333333, 0.333333333333333333333333333333333,
                    0.48968251919873762778370692483619,  0.48968251919873762778370692483619,
                    0.48968251919873762778370692483619,  0.02063496160252474443258615032762,
                    0.02063496160252474443258615032762,  0.48968251919873762778370692483619,
                    0.43708959149293663726993036443535,  0.43708959149293663726993036443535,
                    0.43708959149293663726993036443535,  0.12582081701412672546013927112929,
                    0.12582081701412672546013927112929,  0.43708959149293663726993036443535,
                    0.18820353561903273024096128046733,  0.18820353561903273024096128046733,
                    0.18820353561903273024096128046733,  0.62359292876193453951807743906533,
                    0.62359292876193453951807743906533,  0.18820353561903273024096128046733,
                    0.044729513394452709865106589966276, 0.044729513394452709865106589966276,
                    0.044729513394452709865106589966276, 0.91054097321109458026978682006745,
                    0.91054097321109458026978682006745,  0.044729513394452709865106589966276,
                    0.74119859878449802069007987352342,  0.036838412054736283634817598783385,
                    0.74119859878449802069007987352342,  0.22196298916076569567510252769319,
                    0.036838412054736283634817598783385, 0.74119859878449802069007987352342,
                    0.036838412054736283634817598783385, 0.22196298916076569567510252769319,
                    0.22196298916076569567510252769319,  0.74119859878449802069007987352342,
                    0.22196298916076569567510252769319,  0.036838412054736283634817598783385;

                    weights <<
                    0.097135796282798833819241982507289,
                    0.031334700227139070536854831287209,
                    0.031334700227139070536854831287209,
                    0.031334700227139070536854831287209,
                    0.077827541004774279316739356299404,
                    0.077827541004774279316739356299404,
                    0.077827541004774279316739356299404,
                    0.079647738927210253032891774264045,
                    0.079647738927210253032891774264045,
                    0.079647738927210253032891774264045,
                    0.025577675658698031261678798559000,
                    0.025577675658698031261678798559000,
                    0.025577675658698031261678798559000,
                    0.043283539377289377289377289377289,
                    0.043283539377289377289377289377289,
                    0.043283539377289377289377289377289,
                    0.043283539377289377289377289377289,
                    0.043283539377289377289377289377289,
                    0.043283539377289377289377289377289;

                    break;
                }

                case 12: {
                    points.resize(33, 2);
                    weights.resize(33, 1);

                    points <<
                    0.02356522045239,0.488217389773805,
                    0.488217389773805,0.02356522045239,
                    0.488217389773805,0.488217389773805,
                    0.43972439229446,0.43972439229446,
                    0.43972439229446,0.120551215411079,
                    0.120551215411079,0.43972439229446,
                    0.271210385012116,0.271210385012116,
                    0.271210385012116,0.457579229975768,
                    0.457579229975768,0.271210385012116,
                    0.127576145541586,0.127576145541586,
                    0.127576145541586,0.7448477089168279,
                    0.7448477089168279,0.127576145541586,
                    0.02131735045321,0.02131735045321,
                    0.02131735045321,0.9573652990935799,
                    0.9573652990935799,0.02131735045321,
                    0.115343494534698,0.275713269685514,
                    0.115343494534698,0.6089432357797879,
                    0.275713269685514,0.115343494534698,
                    0.275713269685514,0.6089432357797879,
                    0.6089432357797879,0.115343494534698,
                    0.6089432357797879,0.275713269685514,
                    0.022838332222257,0.28132558098994,
                    0.022838332222257,0.6958360867878031,
                    0.28132558098994,0.022838332222257,
                    0.28132558098994,0.6958360867878031,
                    0.6958360867878031,0.022838332222257,
                    0.6958360867878031,0.28132558098994,
                    0.02573405054833,0.116251915907597,
                    0.02573405054833,0.858014033544073,
                    0.116251915907597,0.02573405054833,
                    0.116251915907597,0.858014033544073,
                    0.858014033544073,0.02573405054833,
                    0.858014033544073,0.116251915907597;

                    weights <<
                    0.025731066440455,
                    0.025731066440455,
                    0.025731066440455,
                    0.043692544538038,
                    0.043692544538038,
                    0.043692544538038,
                    0.062858224217885,
                    0.062858224217885,
                    0.062858224217885,
                    0.034796112930709,
                    0.034796112930709,
                    0.034796112930709,
                    0.006166261051559,
                    0.006166261051559,
                    0.006166261051559,
                    0.040371557766381,
                    0.040371557766381,
                    0.040371557766381,
                    0.040371557766381,
                    0.040371557766381,
                    0.040371557766381,
                    0.022356773202303,
                    0.022356773202303,
                    0.022356773202303,
                    0.022356773202303,
                    0.022356773202303,
                    0.022356773202303,
                    0.017316231108659,
                    0.017316231108659,
                    0.017316231108659,
                    0.017316231108659,
                    0.017316231108659,
                    0.017316231108659;

                    break;
                }
            };
        }
    }

    TriQuadrature::TriQuadrature()
    { }

    void TriQuadrature::get_quadrature(const int order, Quadrature &quad)
    {
        Quadrature tmp;

        get_weight_and_points(order, quad.points, quad.weights);
    }
}
