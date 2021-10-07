// Copyright 2021 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn_wire/server/Server.h"

namespace dawn_wire { namespace server {
    bool Server::DoSetDebugData(uint8_t const * data, uint64_t dataSize) {
        mDebugData.resize(dataSize);
        std::copy(data, data + dataSize, mDebugData.begin());
        return true;
    }

}}  // namespace dawn_wire::server

