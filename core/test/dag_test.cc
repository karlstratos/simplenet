// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../dag.h"

TEST(SharedPointerOwnership, Test) {
  //           A   .D
  //            \. /
  //             .C        Write *X to denote node owned by shared pointer X.
  //            /  \.
  //           B    E
  std::shared_ptr<dag::Node> A = std::make_shared<dag::Node>("A");
  std::shared_ptr<dag::Node> B = std::make_shared<dag::Node>("B");
  {
    std::shared_ptr<dag::Node> C = std::make_shared<dag::Node>("C");
    A->AddChild(C);
    C->AddParent(B);
    {
      std::shared_ptr<dag::Node> D = std::make_shared<dag::Node>("D");
      D->AddParent(C);
      {
        std::shared_ptr<dag::Node> E = std::make_shared<dag::Node>("E");
        C->AddChild(E);
      }
      // *E still has 1 owner: a shared pointer in children_ of *C.
    }
    // *D still has 1 owner: a shared pointer in children_ of *C.
  }
  // *C still has 2 owners:
  //   1. a shared pointer in children_ of *A
  //   2. a shared pointer in children_ of *B

  EXPECT_EQ(1, A.use_count());  // *A currently owned by: A
  EXPECT_EQ(1, B.use_count());  // *B currently owned by: B
  {
    std::shared_ptr<dag::Node> D0 = A->Child(0)->Child(0);
    std::shared_ptr<dag::Node> C0 = D0->Parent(0);
    std::shared_ptr<dag::Node> E0 = C0->Child(1);

    // *D currently owned by: D0, shared pointer in children_ of *C
    EXPECT_EQ(2, D0.use_count());

    // *E currently owned by: E0, shared pointer in children_ of *C
    EXPECT_EQ(2, E0.use_count());

    // *C currently owned by: C0, shared pointer in children_ of *A,
    //                            shared pointer in children_ of *B
    EXPECT_EQ(3, C0.use_count());

    std::shared_ptr<dag::Node> C1 = D0->Parent(0);
    std::shared_ptr<dag::Node> C2 = D0->Parent(0);
    // *C currently owned by: C0, C1, C2, shared pointer in children_ of *A,
    //                                    shared pointer in children_ of *B
    //
    EXPECT_TRUE(C0 == C1);
    EXPECT_TRUE(C1 == C2);
    EXPECT_EQ(5, C0.use_count());
    EXPECT_EQ(5, C1.use_count());
    EXPECT_EQ(5, C2.use_count());
  }
}

TEST(MemoryLeakCheck, Test) {
  //           x   y
  //          / \ / \
  //         l - z - q
  //          \ /__// \
  //           m   /   o
  //            \ /
  //             n
  std::shared_ptr<dag::Node> x = std::make_shared<dag::Node>("x");
  std::shared_ptr<dag::Node> y = std::make_shared<dag::Node>("y");
  std::shared_ptr<dag::Node> z = std::make_shared<dag::Node>("z");
  std::shared_ptr<dag::Node> l = std::make_shared<dag::Node>("l");
  std::shared_ptr<dag::Node> q = std::make_shared<dag::Node>("q");
  std::shared_ptr<dag::Node> m = std::make_shared<dag::Node>("m");
  std::shared_ptr<dag::Node> n = std::make_shared<dag::Node>("n");
  std::shared_ptr<dag::Node> o = std::make_shared<dag::Node>("o");
  x->AddChild(l);
  x->AddChild(z);
  y->AddChild(z);
  y->AddChild(q);
  z->AddChild(l);
  z->AddChild(q);
  l->AddChild(m);
  q->AddChild(n);
  q->AddChild(o);
  m->AddParent(z);
  m->AddParent(q);
  n->AddParent(m);
  EXPECT_EQ("z", x->Child(1)->name());
  EXPECT_EQ("z", y->Child(0)->name());
  EXPECT_EQ("z", l->Parent(1)->name());
  EXPECT_EQ("z", m->Parent(1)->name());
  EXPECT_EQ("z", q->Parent(1)->name());
  EXPECT_EQ("n", o->Parent(0)->Child(0)->name());
}

void expect_node(std::shared_ptr<dag::TreeNode> node, std::string name,
                 size_t num_parents, std::shared_ptr<dag::TreeNode> parent,
                 size_t num_children, int span_begin, int span_end,
                 size_t min_depth, size_t max_depth) {
  EXPECT_EQ(name, node->name());
  EXPECT_EQ(num_parents, node->NumParents());
  EXPECT_EQ(parent, node->Parent());
  EXPECT_EQ(num_children, node->NumChildren());
  EXPECT_EQ(span_begin, node->span_begin());
  EXPECT_EQ(span_end, node->span_end());
  EXPECT_EQ(min_depth, node->min_depth());
  EXPECT_EQ(max_depth, node->max_depth());
}

void expect_preleaf(std::shared_ptr<dag::TreeNode> node, std::string name,
                    size_t num_parents, std::shared_ptr<dag::TreeNode> parent,
                    int span_index, std::string child_name) {
  expect_node(node, name, num_parents, parent, 1, span_index, span_index, 1, 1);
  expect_node(node->Child(0), child_name, 1, node, 0, span_index, span_index, 0,
              0);
}

class TreeReaderTest : public testing::Test {
 protected:
  dag::TreeReader tree_reader_{'(', ')'};
};

TEST_F(TreeReaderTest, ReadGenericTree) {
  //          TOP
  //           |
  //           AA
  //         / |  \
  //      BBB C*#!  D
  //       |   |
  //      bbb  Q
  //           |
  //         *-1-*
  std::shared_ptr<dag::TreeNode> root = tree_reader_.CreateTreeFromTreeString(
      "(TOP(AA   (BBB	bbb)    (C*#! (Q *-1-*  )) D))");

  // nullptr -> [TOP] -> AA
  expect_node(root, "TOP", 0, nullptr, 1, 0, 2, 2, 4);

  // TOP -> [AA] -> BBB C*#1 D
  std::shared_ptr<dag::TreeNode> child1 = root->Child(0);
  expect_node(child1, "AA", 1, root, 3, 0, 2, 1, 3);

  // AA -> [BBB] -> bbb
  std::shared_ptr<dag::TreeNode> child11 = child1->Child(0);
  expect_preleaf(child11, "BBB", 1, child1, 0, "bbb");

  // AA -> [C*#!] -> Q
  std::shared_ptr<dag::TreeNode> child12 = child1->Child(1);
  expect_node(child12, "C*#!", 1, child1, 1, 1, 1, 2, 2);

  // C*#! -> [Q] -> *-1-*
  std::shared_ptr<dag::TreeNode> child121 = child12->Child(0);
  expect_preleaf(child121, "Q", 1, child12, 1, "*-1-*");

  // AA -> [D]
  std::shared_ptr<dag::TreeNode> child13 = child1->Child(2);
  expect_node(child13, "D", 1, child1, 0, 2, 2, 0, 0);
}

TEST_F(TreeReaderTest, ReadDepth1Tree1Child) {
  //     A
  //     |
  //     a
  std::shared_ptr<dag::TreeNode> root \
      = tree_reader_.CreateTreeFromTreeString("(A a)");

  // nullptr -> [A] -> a
  expect_preleaf(root, "A", 0, nullptr, 0, "a");
}

TEST_F(TreeReaderTest, ReadDepth1TreeManyChildren) {
  //     A
  //    /|\
  //   a b c
  std::shared_ptr<dag::TreeNode> root \
      = tree_reader_.CreateTreeFromTreeString("(A a b c)");

  // nullptr -> [A] -> a b c
  expect_node(root, "A", 0, nullptr, 3, 0, 2, 1, 1);

  expect_node(root->Child(0), "a", 1, root, 0, 0, 0, 0, 0);
  expect_node(root->Child(1), "b", 1, root, 0, 1, 1, 0, 0);
  expect_node(root->Child(2), "c", 1, root, 0, 2, 2, 0, 0);
}

TEST_F(TreeReaderTest, ReadSingletonTree) {
  //     a
  std::shared_ptr<dag::TreeNode> root \
      = tree_reader_.CreateTreeFromTreeString("(a)");

  // nullptr -> [a]
  expect_node(root, "a", 0, nullptr, 0, 0, 0, 0, 0);
}

TEST_F(TreeReaderTest, ReadEmptyTree) {
  std::shared_ptr<dag::TreeNode> root \
      = tree_reader_.CreateTreeFromTreeString("()");
  expect_node(root, "", 0, nullptr, 0, 0, 0, 0, 0);

  root = tree_reader_.CreateTreeFromTreeString("((((()))()()))");
  expect_node(root, "", 0, nullptr, 0, 0, 0, 0, 0);
}

TEST_F(TreeReaderTest, ReadTreeWithExtraBrackets) {
  //           /\
  //          A                    A
  //        / | \                  |
  //         / \          ==       B
  //        B                      |
  //        |                      b
  //        b
  std::shared_ptr<dag::TreeNode> root = tree_reader_.CreateTreeFromTreeString(
      "((A () ((B b) ()) ()) ())");

  // nullptr -> [A] -> B
  expect_node(root, "A", 0, nullptr, 1, 0, 0, 2, 2);

  // A -> [B] -> b
  std::shared_ptr<dag::TreeNode> child1 = root->Child(0);
  expect_preleaf(child1, "B", 1, root, 0, "b");
}

TEST_F(TreeReaderTest, Compare) {
  const std::string &tree1_string = "(TOP (A (B b) (C (D d) (E e) (F f))))";
  const std::string &tree2_string = "(TOP(A(B b)(C(D d)(E e)(F f))))";
  const std::string &tree3_string = "(TOP (A (B b) (C (D d) (E z) (F f))))";
  const std::string &tree4_string = "(TOP (A (Q b) (C (D d) (E e) (F f))))";
  const std::string &tree5_string =
      "(TOP (A (B b) (C (D d) (E e) (F f) (G g))))";
  std::shared_ptr<dag::TreeNode> tree1 \
      = tree_reader_.CreateTreeFromTreeString(tree1_string);
  EXPECT_TRUE(tree1->Compare(tree1_string));
  EXPECT_TRUE(tree1->Compare(tree2_string));
  EXPECT_FALSE(tree1->Compare(tree3_string));
  EXPECT_FALSE(tree1->Compare(tree4_string));
  EXPECT_FALSE(tree1->Compare(tree5_string));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
