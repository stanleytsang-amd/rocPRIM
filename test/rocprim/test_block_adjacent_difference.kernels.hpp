// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_
#define TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_

#include "test_utils.hpp"

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize, 1)
void subtract_left_kernel(const T* input, StorageType* output)
{
    /*const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, input + block_offset, thread_items);

    rocprim::block_adjacent_difference<T, BlockSize> adjacent_difference;
    __shared__ typename decltype(adjacent_difference)::storage_type storage; 

    Output thread_output[ItemsPerThread];

    if(blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.subtract_left(
            thread_items, thread_output, BinaryFunction{}, tile_predecessor_item, storage);
    }
    else
    {
        adjacent_difference.subtract_left(thread_items, thread_output, BinaryFunction{}, storage);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, thread_output);*/
}

template <typename T,
          typename Output,
          typename BinaryFunction,
          unsigned int Method,
          unsigned int BlockSize,
          unsigned int ItemsPerThread>
auto test_block_adjacent_difference() -> typename std::enable_if<Method == 3>::type
{
    using stored_type = std::conditional_t<std::is_same<Output, bool>::value, int, Output>;

    static constexpr auto block_size       = BlockSize;
    static constexpr auto items_per_thread = ItemsPerThread;
    static constexpr auto items_per_block  = block_size * items_per_thread;
    static constexpr auto grid_size        = 20;
    static constexpr auto size             = grid_size * items_per_block;

    SCOPED_TRACE(testing::Message() << "with block_size = " << block_size << ", items_per_thread = "
                                    << items_per_thread << ", size = " << size);

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<T>     input = test_utils::get_random_data<T>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        BinaryFunction           op;
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == 0)
                {
                    expected[i] = block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i];
                }
                else
                {
                    expected[i] = op(input[i], input[i - 1]);
                }
            }
        }

        // Preparing Device
        T*           d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(hipMemcpy(
            d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(subtract_left_kernel<T,
                                                                Output,
                                                                stored_type,
                                                                BinaryFunction,
                                                                block_size,
                                                                items_per_thread>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           d_input,
                           d_output);
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(hipMemcpy(
            output.data(), d_output, output.size() * sizeof(output[0]), hipMemcpyDeviceToHost));

        /*ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output, expected, test_utils::precision_threshold<T>::percentage));*/

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}


// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize = 256U
>
struct static_for
{
    static void run()
    {
        int device_id = test_common_utils::obtain_device_from_ctest();
        SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
        HIP_CHECK(hipSetDevice(device_id));

        test_block_adjacent_difference<Type, FlagType, FlagOpType, Method, BlockSize, items[First]>();
        //static_for<First + 1, Last, Type, FlagType, FlagOpType, Method, BlockSize>::run();
    }
};

template <
    unsigned int N,
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int Method,
    unsigned int BlockSize
>
struct static_for<N, N, Type, FlagType, FlagOpType, Method, BlockSize>
{
    static void run()
    {
    }
};

#endif // TEST_BLOCK_ADJACENT_DIFFERENCE_KERNELS_HPP_
