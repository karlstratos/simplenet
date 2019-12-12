// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../neural.h"

TEST(Add, Test0) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;
  double result = z->ForwardBackward();

  EXPECT_EQ(3, result);
  EXPECT_EQ(1, x->get_gradient(0));
  EXPECT_EQ(1, y->get_gradient(0));
}

TEST(Add, Test1) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;  // 1 + 2 = 3
  auto q = z + x;  // 3 + 1 = 4
  auto l = q + q;  // 4 + 4 = 8
  auto o = l + (y + (y + y));  // 4 + (2 + (2 + 2)) = 14
  double result = o->ForwardBackward();

  EXPECT_EQ(14, result);
  EXPECT_EQ(4, x->get_gradient(0));
  EXPECT_EQ(5, y->get_gradient(0));

  EXPECT_EQ(2, model.NumWeights());
  EXPECT_EQ(4, (*model.gradient(0))(0));
  EXPECT_EQ(5, (*model.gradient(1))(0));
}

TEST(Add, Test2) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});

  auto x = model.MakeInput(i_x);
  auto u3 = (x + (x + x)) + x;
  auto y = u3 + (x + (x + x));
  auto t = y + u3;
  auto z = t + y + u3;
  double result = z->ForwardBackward();
  EXPECT_EQ(22, result);
  EXPECT_EQ(22, x->get_gradient(0));
}

TEST(Add, MatrixVector) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 2, 3}});
  size_t i_y = model.AddWeight({{-1}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto z = x + y;
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_EQ(3, result);
  for (size_t i = 0; i < x->NumColumns(); ++i) {
    EXPECT_EQ(1, x->get_gradient(i));
  }
  EXPECT_EQ(3, y->get_gradient(0));
}

TEST(ScalarVariableAddSubtractMultiplyMix, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto y = x + 1;             //     y = x + 1  = 2
  auto z = 1 + 2 * x - 2;     //     z = 2x - 1 = 1
  auto q = y - z;             //     q = -x + 2 = 1
  auto t = z - y;             //     t = x - 2  = -1
  auto l = q * t;             //     l = -x^2 + 4x -4 = -1
  double result = l->ForwardBackward();

  EXPECT_EQ(-1, result);
  EXPECT_EQ(2, x->get_gradient(0));  // -2x + 4
}

TEST(ReduceSum, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 2}, {3, 4}, {5, 6}});
  auto x = model.MakeInput(i_x);
  auto z = sum(x);
  double result = z->ForwardBackward();

  EXPECT_EQ(21, result);
  for (size_t i = 0; i < x->NumRows(); ++i) {
    for (size_t j = 0; j < x->NumColumns(); ++j) {
      EXPECT_EQ(1, x->get_gradient(i, j));
    }
  }
}

TEST(ReduceSumColumnWise, Test) {
  neural::Model model;
  size_t i_X = model.AddWeight({{1, 2}, {3, 4}});
  auto X = model.MakeInput(i_X);
  auto y = sum_cwise(X);  // [4, 6]

  // 16 + 72 = 88
  auto z = column(y, 0) * column(y, 0) + 2 * column(y, 1) * column(y, 1);
  double result = z->ForwardBackward();

  EXPECT_EQ(88, result);
  EXPECT_EQ(8, X->get_gradient(0, 0));
  EXPECT_EQ(8, X->get_gradient(1, 0));
  EXPECT_EQ(24, X->get_gradient(0, 1));
  EXPECT_EQ(24, X->get_gradient(1, 1));
}

TEST(ReduceSumRowWise, Test) {
  neural::Model model;
  size_t i_X = model.AddWeight({{1, 2}, {3, 4}});
  auto X = model.MakeInput(i_X);
  auto y = sum_rwise(X);  // (3, 7)

  // 18 + 49 = 67
  auto z = 2 * row(y, 0) * row(y, 0) + row(y, 1) * row(y, 1);
  double result = z->ForwardBackward();

  EXPECT_EQ(67, result);
  EXPECT_EQ(12, X->get_gradient(0, 0));
  EXPECT_EQ(12, X->get_gradient(0, 1));
  EXPECT_EQ(14, X->get_gradient(1, 0));
  EXPECT_EQ(14, X->get_gradient(1, 1));
}

TEST(ReduceAverageColumnRowWise, Test) {
  neural::Model model;
  size_t i_X = model.AddWeight({{1, 2}, {3, 4}, {5, 6}});
  auto X = model.MakeInput(i_X);
  auto y = average_cwise(X);  // [4, 6]
  auto z = average_rwise(X);  // (1.5, 3.5, 5.5)
  auto y_value = y->Forward();
  auto z_value = z->Forward();

  EXPECT_EQ(3, y_value(0, 0));
  EXPECT_EQ(4, y_value(0, 1));
  EXPECT_EQ(1.5, z_value(0, 0));
  EXPECT_EQ(3.5, z_value(1, 0));
  EXPECT_EQ(5.5, z_value(2, 0));
}

TEST(Multiply, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{2}});
  size_t i_y = model.AddWeight({{3}});

  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto l = x * (y * x) * y;
  auto q = x * l * l * x;  // x^6 y^4
  double result = q->ForwardBackward();

  EXPECT_EQ(5184, result);
  EXPECT_EQ(15552, x->get_gradient(0));  // dq/dx = 6 x^5 y^4
  EXPECT_EQ(6912, y->get_gradient(0));  // dq/dy = 4 x^6 y^3
}

TEST(Dot, Test) {
  neural::Model model;
  size_t i_X = model.AddWeight({{1, 3}, {2, 4}});
  size_t i_Y = model.AddWeight({{5, 7}, {6, 8}});

  auto X = model.MakeInput(i_X);
  auto Y = model.MakeInput(i_Y);
  auto Z = dot(X, Y);
  auto l = sum(Z);
  l->ForwardBackward();

  EXPECT_EQ(17, Z->get_value(0, 0));
  EXPECT_EQ(53, Z->get_value(0, 1));

  EXPECT_EQ(5, X->get_gradient(0, 0));
  EXPECT_EQ(6, X->get_gradient(1, 0));

  EXPECT_EQ(7, X->get_gradient(0, 1));
  EXPECT_EQ(8, X->get_gradient(1, 1));

  EXPECT_EQ(1, Y->get_gradient(0, 0));
  EXPECT_EQ(2, Y->get_gradient(1, 0));

  EXPECT_EQ(3, Y->get_gradient(0, 1));
  EXPECT_EQ(4, Y->get_gradient(1, 1));
}

TEST(Entropy , Test) {
  neural::Model model;
  // ~(1 0 0 0) => entropy 0
  // (1/4 1/4 1/4 1/4) => entropy 2 (log base 2)
  size_t i_X = model.AddWeight({{0.9999999, 0.25}, {0.0000001 / 3.0, 0.25},
                                                   {0.0000001 / 3.0, 0.25},
                                                   {0.0000001 / 3.0, 0.25}});
  auto X = model.MakeInput(i_X);
  auto Z = entropy(X, true);
  sum(Z)->ForwardBackward();
  EXPECT_NEAR(0.0, Z->get_value(0), 1e-4);
  EXPECT_NEAR(2.0, Z->get_value(1), 1e-4);
}

TEST(AddMultiplyDotFlipSign, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}, {2}});
  size_t i_y = model.AddWeight({{3}, {4}});
  auto x = model.MakeInput(i_x);
  auto y = model.MakeInput(i_y);
  auto p = -dot(x + y, 2 * y);  // -2(x'y - y'y)
  double result = p->ForwardBackward();

  EXPECT_EQ(-72, result);

  // Dx = 2y = [6; 8]
  EXPECT_EQ(-6, x->get_gradient(0));
  EXPECT_EQ(-8, x->get_gradient(1));

  // Dy = 2x + 4y = [14; 20]
  EXPECT_EQ(-14, y->get_gradient(0));
  EXPECT_EQ(-20, y->get_gradient(1));
}


TEST(Logistic, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto z = logistic(x / 0.5);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.8808, result, 1e-4);
  EXPECT_NEAR(0.2100, x->get_gradient(0), 1e-4);
}

TEST(Tanh, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  auto x = model.MakeInput(i_x);
  auto z = tanh(2 * x);
  double result = z->ForwardBackward();

  EXPECT_NEAR(0.9640, result, 1e-4);
  EXPECT_NEAR(0.1413, x->get_gradient(0), 1e-4);
}

TEST(Sum, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({{1, 2}, {3, 4}});
  size_t i_X2 = model.AddWeight({{5, 6}, {7, 8}});
  size_t i_X3 = model.AddWeight({{9, 10}, {11, 12}});
  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto X3 = model.MakeInput(i_X3);
  std::vector<std::shared_ptr<neural::Variable>> X123 = {1 * X1, 2 * X2,
                                                         3 * X3};
  auto X = sum(X123);
  sum(X)->ForwardBackward();

  EXPECT_EQ(1, X1->get_gradient(0, 0));
  EXPECT_EQ(2, X2->get_gradient(0, 0));
  EXPECT_EQ(3, X3->get_gradient(0, 0));
}

TEST(ConcatenateVertical, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  size_t i_X2 = model.AddWeight({{9, 10}, {11, 12}});
  size_t i_X3 = model.AddWeight({{13, 14}, {15, 16}, {17, 18}});
  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto X3 = model.MakeInput(i_X3);
  std::vector<std::shared_ptr<neural::Variable>> X123 = {X1, X2, X3};
  auto X = vcat(X123);
  sum((X % X)/2)->ForwardBackward();

  EXPECT_EQ(1, X1->get_gradient(0, 0));
  EXPECT_EQ(12, X2->get_gradient(1, 1));
  EXPECT_EQ(17, X3->get_gradient(2, 0));
}

TEST(ConcatenateHorizontal, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({{1}, {2}});
  size_t i_X2 = model.AddWeight({{3, 4}, {5, 6}});
  size_t i_X3 = model.AddWeight({{7}, {9}});
  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto X3 = model.MakeInput(i_X3);
  std::vector<std::shared_ptr<neural::Variable>> X123 = {X1, X2, X3};
  auto X = hcat(X123);
  sum((X % X)/2)->ForwardBackward();

  EXPECT_EQ(1, X1->get_gradient(0, 0));
  EXPECT_EQ(6, X2->get_gradient(1, 1));
  EXPECT_EQ(9, X3->get_gradient(1, 0));
}

TEST(SoftmaxPick, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 1}, {2, 2}, {3, 5}});
  auto x = model.MakeInput(i_x);

  // y = [0.0900   0.0171
  //      0.2447   0.0466
  //      0.6652   0.9362]
  auto y = softmax(x);

  // z = [0.2447   0.9362]
  auto z = pick(y, {1, 2});

  // l = 1.1809
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(1.1809, result, 1e-4);
  EXPECT_NEAR(0.2447, z->get_value(0), 1e-4);
  EXPECT_NEAR(0.9362, z->get_value(1), 1e-4);

  EXPECT_NEAR(0.0900, y->get_value(0, 0), 1e-4);
  EXPECT_NEAR(0.2447, y->get_value(1, 0), 1e-4);
  EXPECT_NEAR(0.6652, y->get_value(2, 0), 1e-4);
  EXPECT_NEAR(0.0171, y->get_value(0, 1), 1e-4);
  EXPECT_NEAR(0.0466, y->get_value(1, 1), 1e-4);
  EXPECT_NEAR(0.9362, y->get_value(2, 1), 1e-4);

  EXPECT_NEAR(-0.0220, x->get_gradient(0, 0), 1e-4);
  EXPECT_NEAR(0.1848, x->get_gradient(1, 0), 1e-4);
  EXPECT_NEAR(-0.1628, x->get_gradient(2, 0), 1e-4);
  EXPECT_NEAR(-0.0160, x->get_gradient(0, 1), 1e-4);
  EXPECT_NEAR(-0.0436, x->get_gradient(1, 1), 1e-4);
  EXPECT_NEAR(0.0597, x->get_gradient(2, 1), 1e-4);
}

TEST(PickNegativeLogSoftmax, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1, 1}, {2, 2}, {3, 5}});
  auto x = model.MakeInput(i_x);

  // [0.0900   0.0171
  //  0.2447   0.0466     =>   -log(0.2447..) - log(0.9362..) = 1.4735
  //  0.6652   0.9362]
  auto l = sum(cross_entropy(x, {1, 2}));
  l->ForwardBackward();

  EXPECT_NEAR(1.4735, l->get_value(0), 1e-4);

  EXPECT_NEAR(0.0900, x->get_gradient(0, 0), 1e-4);  // p(0)
  EXPECT_NEAR(-0.7553, x->get_gradient(1, 0), 1e-4); // p(1) - 1
  EXPECT_NEAR(0.6652, x->get_gradient(2, 0), 1e-4);  // p(2)

  EXPECT_NEAR(0.0171, x->get_gradient(0, 1), 1e-4);  // p(0)
  EXPECT_NEAR(0.0466, x->get_gradient(1, 1), 1e-4);  // p(1)
  EXPECT_NEAR(-0.0638, x->get_gradient(2, 1), 1e-4); // p(2) - 1
}

TEST(FlagNegativeLogistic, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{-1, 2}});

  auto x = model.MakeInput(i_x);
  auto z = binary_cross_entropy(x, {false, true});
  auto l = sum(z);
  double result = l->ForwardBackward();

  EXPECT_NEAR(0.4402, result, 1e-4);  // -log p(F1) - log p(T2)
  EXPECT_NEAR(0.3133, z->get_value(0), 1e-4);  // -log p(F1)
  EXPECT_NEAR(0.1269, z->get_value(1), 1e-4);  // -log p(T2)
  EXPECT_NEAR(0.2689, x->get_gradient(0), 1e-4);  //  p(T1)
  EXPECT_NEAR(-0.1192, x->get_gradient(1), 1e-4); // -p(F2)
}

TEST(Feedforward, GradientCheck) {
  std::srand(std::time(0));
  double epsilon = 1e-4;
  size_t num_examples = 5;
  size_t dim_input = 10;
  size_t num_labels = 3;

  std::vector<size_t> labels;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dis(0, num_labels - 1);
  for (size_t i = 0; i < num_examples; ++i) { labels.push_back(dis(gen)); }

  neural::Model model;
  size_t i_X = model.AddWeight(dim_input, num_examples, "unit-variance");
  size_t i_W1 = model.AddWeight(num_labels, dim_input, "unit-variance");
  size_t i_b1 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_W2 = model.AddWeight(num_labels, num_labels, "unit-variance");
  size_t i_b2 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_W3 = model.AddWeight(num_labels, num_labels, "unit-variance");
  size_t i_b3 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_W4 = model.AddWeight(num_labels, num_labels, "unit-variance");
  size_t i_b4 = model.AddWeight(num_labels, 1, "unit-variance");
  size_t i_u = model.AddWeight(num_labels, num_examples, "unit-variance");

  auto compute_output = [&](Eigen::MatrixXd *X_grad) {
    auto X = model.MakeInput(i_X);
    auto W1 = model.MakeInput(i_W1);
    auto b1 = model.MakeInput(i_b1);
    auto W2 = model.MakeInput(i_W2);
    auto b2 = model.MakeInput(i_b2);
    auto W3 = model.MakeInput(i_W3);
    auto b3 = model.MakeInput(i_b3);
    auto W4 = model.MakeInput(i_W4);
    auto b4 = model.MakeInput(i_b4);
    auto u = model.MakeInput(i_u);
    auto H = logistic(W4 * (W3 * relu(W2 * tanh(W1 * X + b1) + b2) + b3)
                      + b4) % u / 10.0;
    auto l = average(cross_entropy(H, labels));
    double l_value = l->ForwardBackward();
    if (X_grad != nullptr) { *X_grad = X->ref_gradient(); }
    return l_value;
  };

  Eigen::MatrixXd X_grad;
  double l0 = compute_output(&X_grad);

  for (size_t i = 0; i < X_grad.rows(); ++i) {
    for (size_t j = 0; j < X_grad.cols(); ++j) {
      (*model.weight(i_X))(i, j) += epsilon;
      double l1 = compute_output(nullptr);
      EXPECT_NEAR((l1 - l0) / epsilon, X_grad(i, j), 1e-5);
      (*model.weight(i_X))(i, j) -= epsilon;
    }
  }
}

TEST(Adam, Test) {
  neural::Model model;
  size_t i_X = model.AddWeight({{0}});
  auto X = model.MakeInput(i_X);
  X->ref_gradient()(0, 0) = 10;  // X already has gradient of shape (1, 1).

  double step_size = 0.5;
  double b1 = 0.6;
  double b2 = 0.3;
  double ep = 0.1;
  neural::Adam gd(&model, step_size, b1, b2, ep);
  gd.UpdateWeights();

  EXPECT_NEAR(X->get_value(0), -0.4941, 1e-4);
}

TEST(SimpleRNN, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({{1, 3}, {2, 4}});
  size_t i_X2 = model.AddWeight({{0, 0}, {1, -1}});
  neural::SimpleRNN srnn(2, 2, 1, &model);
  Eigen::MatrixXd U1(1, 2);
  U1 << 1, 1;
  Eigen::MatrixXd U2(1, 1);
  U2 << 2;
  Eigen::MatrixXd V1(1, 1);
  V1 << 3;
  Eigen::MatrixXd V2(1, 1);
  V2 << 2;
  Eigen::MatrixXd b1(1, 1);
  b1 << 1;
  Eigen::MatrixXd b2(1, 1);
  b2 << -1;
  srnn.SetWeights(U1, V1, b1, 0);
  srnn.SetWeights(U2, V2, b2, 1);

  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto HHs = srnn.Transduce({X1, X2});
  Eigen::MatrixXd upper_right_H = HHs.back().back()[0]->Forward();

  EXPECT_NEAR(upper_right_H(0, 0), 0.9872, 1e-4);
  EXPECT_NEAR(upper_right_H(0, 1), 0.9870, 1e-4);
}

TEST(LSTM, Test) {
  neural::Model model;
  size_t i_X1 = model.AddWeight({
      {-0.769665913753682, 1.117356138814467},
      {0.371378812760058, -1.089064295052236},
      {-0.225584402271252, 0.032557464164973}});
  size_t i_X2 = model.AddWeight({
      {0.552527021112224, 0.085931133175425},
      {1.100610217880866, -1.491590310637609},
      {1.544211895503951, -0.742301837259857}});

  neural::LSTM lstm(2, 3, 2, &model);
  Eigen::MatrixXd U1(8, 3);
  Eigen::MatrixXd V1(8, 2);
  Eigen::MatrixXd b1(8, 1);
  U1 <<
      -1.061581733319986, -1.422375925091496, -0.804465956349547,
      2.350457224002042, 0.488193909859941, 0.696624415849607,
      -0.615601881466894, -0.177375156618825, 0.835088165072682,
      0.748076783703985, -0.196053487807333, -0.243715140377952,
      -0.192418510588264, 1.419310150642549, 0.215670086403744,
      0.888610425420721, 0.291584373984183, -1.165843931482049,
      -0.764849236567874, 0.197811053464361, -1.147952778898594,
      -1.402268969338759, 1.587699089974059, 0.104874716016494;
  V1 <<
      0.722254032225002, 0.840375529753905,
      2.585491252616241, -0.888032082329010,
      -0.666890670701386, 0.100092833139322,
      0.187331024578940, -0.544528929990548,
      -0.082494425370955, 0.303520794649354,
      -1.933022917850987, -0.600326562133734,
      -0.438966153934773, 0.489965321173948,
      -1.794678841455123, 0.739363123604474;
  b1 <<
      1.436696622718939,
      -1.960899999365033,
      -0.197698225974150,
      -1.207845485259799,
      2.908008030729362,
      0.825218894228491,
      1.378971977916614,
      -1.058180257987362;

  Eigen::MatrixXd U2(8, 2);
  Eigen::MatrixXd V2(8, 2);
  Eigen::MatrixXd b2(8, 1);
  U2 <<
      -0.468615581100624,  -1.577057022799202,
      -0.272469409250187,   0.507974650905946,
      1.098424617888623,   0.281984063670556,
      -0.277871932787639,   0.033479882244451,
      0.701541458163284,  -1.333677943428106,
      -2.051816299911149,   1.127492278341590,
      -0.353849997774433,   0.350179410603312,
      -0.823586525156853,  -0.299066030332982;
  V2 <<
      0.022889792751630,  -2.002635735883060,
      -0.261995434966092,   0.964229422631627,
      -1.750212368446790,   0.520060101455458,
      -0.285650971595330,  -0.020027851642538,
      -0.831366511567624,  -0.034771086028483,
      -0.979206305167302,  -0.798163584564142,
      -1.156401655664002,   1.018685282128575,
      -0.533557109315987,  -0.133217479507735;
  b2 <<
      -0.714530163787158,
      1.351385768426657,
      -0.224771056052584,
      -0.589029030720801,
      -0.293753597735416,
      -0.847926243637934,
      -1.120128301243728,
      2.525999692118309;

  lstm.SetWeights(U1, V1, b1, 0);
  lstm.SetWeights(U2, V2, b2, 1);
  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto HHs = lstm.Transduce({X1, X2});

  Eigen::MatrixXd H1_top = HHs[0].back()[0]->Forward();
  EXPECT_NEAR(H1_top(0, 0), -0.072877305703618, 1e-15);
  EXPECT_NEAR(H1_top(0, 1), -0.073987670207449, 1e-15);
  EXPECT_NEAR(H1_top(1, 0), 0.241931100637694, 1e-15);
  EXPECT_NEAR(H1_top(1, 1), 0.265201583136943, 1e-15);

  Eigen::MatrixXd H2_top = HHs[1].back()[0]->Forward();
  EXPECT_NEAR(H2_top(0, 0), -0.166538899924138, 1e-15);
  EXPECT_NEAR(H2_top(0, 1), -0.177181516362888, 1e-15);
  EXPECT_NEAR(H2_top(1, 0), 0.365075202550646, 1e-15);
  EXPECT_NEAR(H2_top(1, 1), 0.297207356699930, 1e-15);
}

TEST(LSTM, GradientCheck) {
  std::srand(std::time(0));
  double epsilon = 1e-4;
  size_t num_labels = 3;
  size_t batch_size = 4;
  size_t dim_observation = 2;
  size_t dim_state = 3;
  size_t num_layers = 3;

  std::vector<size_t> labels;
  std::default_random_engine gen;
  std::uniform_int_distribution<size_t> dis(0, num_labels - 1);
  for (size_t i = 0; i < batch_size; ++i) { labels.push_back(dis(gen)); }

  neural::Model model;
  size_t i_X1 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  size_t i_X2 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  size_t i_X3 = model.AddWeight(dim_observation, batch_size, "unit-variance");
  neural::LSTM lstm(num_layers, dim_observation, dim_state, &model);
  size_t i_W = model.AddWeight(num_labels, dim_state, "unit-variance");

  auto compute_output = [&](Eigen::MatrixXd *X1_grad) {
    auto X1 = model.MakeInput(i_X1);
    auto X2 = model.MakeInput(i_X2);
    auto X3 = model.MakeInput(i_X3);
    auto H = lstm.Transduce({X1, X2, X3}).back().back()[0];
    auto W = model.MakeInput(i_W);
    auto l = average(cross_entropy(W * H, labels));
    double l_value = l->ForwardBackward();
    if (X1_grad != nullptr) { *X1_grad = X1->ref_gradient(); }
    return l_value;
  };

  Eigen::MatrixXd X1_grad;
  double l0 = compute_output(&X1_grad);

  for (size_t i = 0; i < X1_grad.rows(); ++i) {
    for (size_t j = 0; j < X1_grad.cols(); ++j) {
      (*model.weight(i_X1))(i, j) += epsilon;
      double l1 = compute_output(nullptr);
      EXPECT_NEAR((l1 - l0) / epsilon, X1_grad(i, j), 1e-5);
      (*model.weight(i_X1))(i, j) -= epsilon;
    }
  }
}

TEST(LSTM, DropoutDoesNotCrash) {
  neural::Model model;
  size_t i_X1 = model.AddWeight(5, 10, "unit-variance");
  size_t i_X2 = model.AddWeight(5, 10, "unit-variance");
  neural::LSTM lstm(2, 5, 20, &model);
  lstm.UseDropout(0.5, 42);

  auto X1 = model.MakeInput(i_X1);
  auto X2 = model.MakeInput(i_X2);
  auto HHs = lstm.Transduce({X1, X2});
  sum(HHs.back().back()[0])->ForwardBackward();
}

TEST(LSTM, BatchPadding) {
  neural::Model model;
  size_t i_x1 = model.AddWeight({{1}, {1}});
  size_t i_x2 = model.AddWeight({{2}, {2}});
  size_t i_x3 = model.AddWeight({{3}, {3}});
  neural::LSTM lstm(3, 2, 20, &model);

  auto x1 = model.MakeInput(i_x1);
  auto x2 = model.MakeInput(i_x2);
  auto x3 = model.MakeInput(i_x3);

  // 1 3 2     2 1 0     3 2 0     0 3 0     0 1 0
  // 1 3 2     2 1 0     3 2 0     0 3 0     0 1 0
  auto Xs = lstm.Batch({{x1, x2, x3}, {x3, x1, x2, x3, x1}, {x2}});
  auto HHs = lstm.Transduce(Xs);
  sum(HHs.back().back()[0])->ForwardBackward();

  EXPECT_EQ(1, Xs[0]->get_value(0, 0));
  EXPECT_EQ(1, Xs[0]->get_value(1, 0));
  EXPECT_EQ(3, Xs[0]->get_value(0, 1));
  EXPECT_EQ(3, Xs[0]->get_value(1, 1));
  EXPECT_EQ(2, Xs[0]->get_value(0, 2));
  EXPECT_EQ(2, Xs[0]->get_value(1, 2));

  EXPECT_EQ(0, Xs[4]->get_value(0, 0));
  EXPECT_EQ(0, Xs[4]->get_value(1, 0));
  EXPECT_EQ(1, Xs[4]->get_value(0, 1));
  EXPECT_EQ(1, Xs[4]->get_value(1, 1));
  EXPECT_EQ(0, Xs[4]->get_value(0, 2));
  EXPECT_EQ(0, Xs[4]->get_value(1, 2));
}

TEST(OverwriteSharedPointers, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  const auto &x = model.MakeInput(i_x);
  const auto &y = model.MakeInput(i_y);
  auto z = x + y;  // 1 + 2
  z = z + y;  // 3 + 2
  z = z + x;  // 5 + 1
  z = z * y;  // 6 * 2
  double result = z->ForwardBackward();

  EXPECT_EQ(12, result);  // 2xy + 2y^2
  EXPECT_EQ(4, x->get_gradient(0));  // 2y
  EXPECT_EQ(10, y->get_gradient(0));  // 2x + 4y
}

TEST(IntermediateForwardCalls, Test) {
  neural::Model model;
  size_t i_x = model.AddWeight({{1}});
  size_t i_y = model.AddWeight({{2}});

  const auto &x = model.MakeInput(i_x);
  const auto &y = model.MakeInput(i_y);
  auto z = x + y;  // 1 + 2
  EXPECT_EQ(3, z->Forward()(0, 0));

  z = z + y;  // 3 + 2
  EXPECT_EQ(5, z->Forward()(0, 0));

  z = z + x;  // 5 + 1
  EXPECT_EQ(6, z->Forward()(0, 0));

  z = z * y;  // 6 * 2
  EXPECT_EQ(12, z->Forward()(0, 0));

  double result = z->ForwardBackward();
  EXPECT_EQ(12, result);  // 2xy + 2y^2
  EXPECT_EQ(4, x->get_gradient(0));  // 2y
  EXPECT_EQ(10, y->get_gradient(0));  // 2x + 4y
}

TEST(InputColumn, OnlyIndividualColumnUpdates) {
  neural::Model model;
  //   1    2    3    4
  //   1    2    3    4
  //   1    2    3    4
  size_t i_X = model.AddWeight({{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}});
  neural::SimpleGradientDescent gd(&model, 0.01);

  auto x2 = model.MakeInputColumn(i_X, 1);  // (2 2 2)
  auto x3 = model.MakeInputColumn(i_X, 2);  // (3 3 3)
  EXPECT_EQ(3, x2->NumRows());
  EXPECT_EQ(1, x2->NumColumns());

  auto y = x2 % x3;  // (6 6 6)
  y = sum(y);  // 18
  double result = y->ForwardBackward();

  gd.UpdateWeights();

  //   1    1.97    2.98    4
  //   1    1.97    2.98    4
  //   1    1.97    2.98    4
  EXPECT_EQ(18, result);
  EXPECT_EQ(1.97, (*model.weight(i_X))(0, 1));
  EXPECT_EQ(1.97, (*model.weight(i_X))(1, 1));
  EXPECT_EQ(1.97, (*model.weight(i_X))(2, 1));
  EXPECT_EQ(2.98, (*model.weight(i_X))(0, 2));
  EXPECT_EQ(2.98, (*model.weight(i_X))(1, 2));
  EXPECT_EQ(2.98, (*model.weight(i_X))(2, 2));

  EXPECT_EQ(0, gd.num_updates(i_X));
  EXPECT_EQ(0, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));
  EXPECT_EQ(0, gd.num_column_updates(i_X, 3));

  auto x4 = model.MakeInputColumn(i_X, 3);  // (4 4 4)
  x2 = model.MakeInputColumn(i_X, 1);  // (1.97 1.97 1.97)
  y = x4 + x2;  // (5.97 5.97 5.97)
  y = sum(y);  // 17.9
  result = y->ForwardBackward();
  gd.UpdateWeights();

  //   1    1.96    2.98    3.99
  //   1    1.96    2.98    3.99
  //   1    1.96    2.98    3.99
  EXPECT_EQ(17.91, result);
  EXPECT_EQ(1.96, (*model.weight(i_X))(0, 1));
  EXPECT_EQ(1.96, (*model.weight(i_X))(1, 1));
  EXPECT_EQ(1.96, (*model.weight(i_X))(2, 1));
  EXPECT_EQ(3.99, (*model.weight(i_X))(0, 3));
  EXPECT_EQ(3.99, (*model.weight(i_X))(1, 3));
  EXPECT_EQ(3.99, (*model.weight(i_X))(2, 3));

  EXPECT_EQ(0, gd.num_updates(i_X));
  EXPECT_EQ(0, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(2, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 3));
}

TEST(InputColumn, MixedUpdates) {
  neural::Model model;
  //   1    2    3
  size_t i_X = model.AddWeight({{1, 2, 3}});
  neural::SimpleGradientDescent gd(&model, 0.01);

  auto x1 = model.MakeInputColumn(i_X, 0);  // 1
  auto x3 = model.MakeInputColumn(i_X, 2);  // 3
  auto X = model.MakeInput(i_X);            // [1 2 3]
  auto y = sum((x1 % x3) * X);              // sum(3 * [1 2 3]) = 18
  double result = y->ForwardBackward();

  // x1^2 x3 + x_1 x2 x3 + x1 x3^2
  // d(x1) = 2 x1 x3 + x2 x3 + x3^2 = 6 + 6 + 9 = 21
  // d(x2) = x1 x3 = 3
  // d(x3) = x1^2 + x1 x2 + 2 x1 x3 = 1 + 2 + 6 = 9
  EXPECT_EQ(18, result);
  EXPECT_EQ(21, (*model.gradient(i_X))(0));
  EXPECT_EQ(3, (*model.gradient(i_X))(1));
  EXPECT_EQ(9, (*model.gradient(i_X))(2));

  gd.UpdateWeights();
  EXPECT_EQ(0.79, (*model.weight(i_X))(0));
  EXPECT_EQ(1.97, (*model.weight(i_X))(1));
  EXPECT_EQ(2.91, (*model.weight(i_X))(2));

  EXPECT_EQ(1, gd.num_updates(i_X));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));

  x1 = model.MakeInputColumn(i_X, 0);
  result = x1->ForwardBackward();
  EXPECT_EQ(0.79, result);
  EXPECT_EQ(1, (*model.gradient(i_X))(0));

  gd.UpdateWeights();
  EXPECT_EQ(0.78, (*model.weight(i_X))(0));
  EXPECT_EQ(1.97, (*model.weight(i_X))(1));
  EXPECT_EQ(2.91, (*model.weight(i_X))(2));

  EXPECT_EQ(1, gd.num_updates(i_X));
  EXPECT_EQ(2, gd.num_column_updates(i_X, 0));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 1));
  EXPECT_EQ(1, gd.num_column_updates(i_X, 2));
}

TEST(Block, Test) {
  neural::Model model;
  // -2 -1  0
  //  1  2  3          1  2  3
  //  4  5  6    =>    4  5  6      =>   5  6
  //  7  8  9          7  8  9           8  9
  // 10 11 12
  size_t i_X = model.AddWeight({{-2, -1, 0},
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
  auto X = model.MakeInput(i_X);
  auto Y = rows(X, 1, 3);  // [1 2 3; 4 5 6; 7 8 9]
  auto Z = sum(block(Y, 1, 1, 2, 2));  // sum([5 6; 8 9])
  double result = Z->ForwardBackward();
  EXPECT_EQ(28, result);
  //  0  0  0
  //  0  0  0
  //  0  1  1
  //  0  1  1
  //  0  0  0
  EXPECT_EQ(0, X->get_gradient(0, 0));
  EXPECT_EQ(0, X->get_gradient(0, 1));
  EXPECT_EQ(0, X->get_gradient(0, 2));
  EXPECT_EQ(0, X->get_gradient(1, 0));
  EXPECT_EQ(0, X->get_gradient(1, 1));
  EXPECT_EQ(0, X->get_gradient(1, 2));
  EXPECT_EQ(0, X->get_gradient(2, 0));
  EXPECT_EQ(1, X->get_gradient(2, 1));
  EXPECT_EQ(1, X->get_gradient(2, 2));
  EXPECT_EQ(0, X->get_gradient(3, 0));
  EXPECT_EQ(1, X->get_gradient(3, 1));
  EXPECT_EQ(1, X->get_gradient(3, 2));
  EXPECT_EQ(0, X->get_gradient(4, 0));
  EXPECT_EQ(0, X->get_gradient(4, 1));
  EXPECT_EQ(0, X->get_gradient(4, 2));
}

TEST(RNN, EncodeByFinalTop) {
  neural::Model model;
  size_t i_a = model.AddWeight(10, 1, "unit-variance");
  size_t i_b = model.AddWeight(10, 1, "unit-variance");
  neural::LSTM lstm(2, 10, 10, &model);

  double result_value = 0.0;
  Eigen::MatrixXd a_gradient;
  Eigen::MatrixXd b_gradient;

  // Run LSTM on each sequence separately and use the final top hidden states.
  {
    auto a = model.MakeInput(i_a);
    auto b = model.MakeInput(i_b);
    std::vector<std::shared_ptr<neural::Variable>> seq1 = {a, a, b};
    std::vector<std::shared_ptr<neural::Variable>> seq2 = {a, a};
    std::vector<std::shared_ptr<neural::Variable>> seq3 = {a, a, b, b};
    std::vector<std::shared_ptr<neural::Variable>> seq4 = {b};
    auto h1 = lstm.Transduce(seq1).back().back()[0];
    auto h2 = lstm.Transduce(seq2).back().back()[0];
    auto h3 = lstm.Transduce(seq3).back().back()[0];
    auto h4 = lstm.Transduce(seq4).back().back()[0];
    auto hs = {h1, h2, h3, h4};
    result_value = sum(sum(hs))->ForwardBackward();
    a_gradient = a->ref_gradient();
    b_gradient = b->ref_gradient();
    model.ClearComputation();
  }

  // Run LSTM on batch.
  auto a = model.MakeInput(i_a);
  auto b = model.MakeInput(i_b);
  std::vector<std::shared_ptr<neural::Variable>> seq1 = {a, a, b};
  std::vector<std::shared_ptr<neural::Variable>> seq2 = {a, a};
  std::vector<std::shared_ptr<neural::Variable>> seq3 = {a, a, b, b};
  std::vector<std::shared_ptr<neural::Variable>> seq4 = {b};
  auto seqs = {seq1, seq2, seq3, seq4};
  auto hs = lstm.EncodeByFinalTop(seqs);
  double this_result_value = sum(sum(hs))->ForwardBackward();
  a_gradient = a->ref_gradient();
  Eigen::MatrixXd this_a_gradient = a->ref_gradient();
  Eigen::MatrixXd this_b_gradient = b->ref_gradient();

  EXPECT_NEAR(result_value, this_result_value, 1e-10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(a_gradient(i), this_a_gradient(i), 1e-10);
    EXPECT_NEAR(b_gradient(i), this_b_gradient(i), 1e-10);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
