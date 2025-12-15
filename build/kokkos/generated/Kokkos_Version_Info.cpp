// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "Kokkos_Version_Info.hpp"

namespace Kokkos {
namespace Impl {

std::string GIT_BRANCH       = R"branch(develop)branch";
std::string GIT_COMMIT_HASH  = "92c1c2337";
std::string GIT_CLEAN_STATUS = "DIRTY";
std::string GIT_COMMIT_DESCRIPTION =
    R"message(remove old <c++20 guard)message";
std::string GIT_COMMIT_DATE = "2025-12-11T12:48:57-07:00";

}  // namespace Impl

}  // namespace Kokkos
