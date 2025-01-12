project('cuda_debayer', ['cpp','cuda'],
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++23', 'b_asneeded=false'])
cuda = import('unstable-cuda')

# arch_flags = cuda.nvcc_arch_flags('Ampere', detected:['8.0 8.6'])
# arch_flags = cuda.nvcc_arch_flags('Ampere', detected:['8.6+PTX'])
arch_flags = ['-gencode', 'arch=compute_86,code=sm_86']
message(arch_flags)

include_dirs = include_directories('include')

cuda_dep = dependency('cuda', version : '>=11')
ccfits_dep = dependency('CCfits', static:true)
cfitsio_dep = dependency('cfitsio', static:true)

deps = [cuda_dep, ccfits_dep, cfitsio_dep]

source_files = ['cuda_debayer.cpp', 'kernel.cu']

executable('cuda_debayer',
           source_files,
           include_directories: include_dirs,
           install : true,
           dependencies: [deps],
           cuda_args: arch_flags,
           link_args:['-lcurl'],
           )
