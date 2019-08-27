// Copyright 2019 The Dawn Authors
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

#ifndef DAWNNATIVE_D3D12_COMMITTEDRESOURCEALLOCATORD3D12_H_
#define DAWNNATIVE_D3D12_COMMITTEDRESOURCEALLOCATORD3D12_H_

#include "common/SerialQueue.h"
#include "dawn_native/Error.h"
#include "dawn_native/ResourceMemoryAllocation.h"
#include "dawn_native/d3d12/d3d12_platform.h"

namespace dawn_native { namespace d3d12 {

    class Device;

    // Wrapper to allocate D3D12 committed resource.
    // Committed resources are implicitly backed by a D3D12 heap.
    class CommittedResourceAllocator {
      public:
        CommittedResourceAllocator(Device* device, D3D12_HEAP_TYPE heapType);
        ~CommittedResourceAllocator() = default;

        ResultOrError<ResourceMemoryAllocation> Allocate(
            const D3D12_RESOURCE_DESC& resourceDescriptor,
            D3D12_RESOURCE_STATES initialUsage,
            D3D12_HEAP_FLAGS heapFlags);
        void Deallocate(ResourceMemoryAllocation& allocation);

      private:
        Device* mDevice;
        D3D12_HEAP_TYPE mHeapType;
    };

}}  // namespace dawn_native::d3d12

#endif  // DAWNNATIVE_D3D12_COMMITTEDRESOURCEALLOCATORD3D12_H_
