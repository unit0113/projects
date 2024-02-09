#include "gtest/gtest.h"
#include "Divisibility.h"

//Unit tests go here
TEST(test1, Apos_Bpos_divisible) {
	ASSERT_EQ(isDivisible(2, 4), true);
}

TEST(test2, Apos_Bpos_indivisible) {
	ASSERT_EQ(isDivisible(2, 3), false);
}

TEST(test3, Apos_B0) {
	ASSERT_EQ(isDivisible(2, 0), true);
}

TEST(test4, Aneg_Bpos_divisible) {
	ASSERT_EQ(isDivisible(-2, 4), true);
}
