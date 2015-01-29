^int s0;$
^const int s1 = 4;$
^unsigned int s2;$

^edu::guard::ptr_guard_t<int> p0;$
^edu::guard::ptr_guard_t<const int> p1 = &s1;$


^edu::guard::array1_guard_t<int, 5> a1;$
^edu::guard::array2_guard_t<int, 10, 20> a2;$

^edu::guard::ptr_guard_t<int> es0 = (int\*)__edu_cuda_get_dynamic_shared();$
^edu::guard::ptr_guard_t<int> es1 = (int\*)__edu_cuda_get_dynamic_shared();$

^__edu_cuda_shared_storage edu::guard::array1_guard_t<int, 5> __edu_cuda_shared_ss0;edu::guard::array1_guard_t<int, 5> &ss0 = __edu_cuda_shared_ss0;$