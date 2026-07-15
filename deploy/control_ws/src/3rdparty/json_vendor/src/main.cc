//
// Woan Technology (Shenzhen) Co., Ltd. CONFIDENTIAL
//
// Unpublished Copyright (c) 2018 - 2022
// [Woan Technology (Shenzhen) Co., Ltd.], All Rights Reserved.
//
// Author: Albert Lv <lvwenyi@wondertechlabs.com>
//

#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

int main() {
  nlohmann::json json = {{"pi", 3.141},
                         {"happy", true},
                         {"name", "Niels"},
                         {"nothing", nullptr},
                         {"answer", {{"everything", 42}}},
                         {"list", {1, 0, 2}},
                         {"object", {{"currency", "USD"}, {"value", 42.99}}}};

  std::cout << "declared JSON object: " << std::setw(2) << json << std::endl;

  return 0;
}
