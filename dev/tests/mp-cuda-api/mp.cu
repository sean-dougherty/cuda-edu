// Sanity check that we can compile and execute a Cuda program.

// MP 1
#include <wb.h>
#include <assert.h>

using namespace std;

void test_vector_types() {
    {
        float1 f1 = make_float1(1.0f);
        assert(f1.x == 1.0f);

        float2 f2 = make_float2(1.0f, 2.0f);
        assert(f2.x == 1.0f);
        assert(f2.y == 2.0f);

        float3 f3 = make_float3(1.0f, 2.0f, 3.0f);
        assert(f3.x == 1.0f);
        assert(f3.y == 2.0f);
        assert(f3.z == 3.0f);

        float4 f4 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        assert(f4.x == 1.0f);
        assert(f4.y == 2.0f);
        assert(f4.z == 3.0f);
        assert(f4.w == 4.0f);
    }

    {
        int1 i1 = make_int1(1.0f);
        assert(i1.x == 1.0f);

        int2 i2 = make_int2(1.0f, 2.0f);
        assert(i2.x == 1.0f);
        assert(i2.y == 2.0f);

        int3 i3 = make_int3(1.0f, 2.0f, 3.0f);
        assert(i3.x == 1.0f);
        assert(i3.y == 2.0f);
        assert(i3.z == 3.0f);

        int4 i4 = make_int4(1.0f, 2.0f, 3.0f, 4.0f);
        assert(i4.x == 1.0f);
        assert(i4.y == 2.0f);
        assert(i4.z == 3.0f);
        assert(i4.w == 4.0f);
    }
}

int main(int argc, char **argv) {
    test_vector_types();

    cout << "--- Passed cuda-api tests ---" << endl;
    return 0;
}
