# This module finds an installed armadilo package.
#
# It sets the following variables:
#  Opengm_FOUND              - Set to false, or undefined, if lemon isn't found.
#  Opengm_INCLUDE_DIR        - Lemon include directory.
#  Opengm_LIBRARIES          - Lemon library files
FIND_PATH(OpenGM_INCLUDE_DIR opengm PATHS /usr/include /usr/local/include ${CMAKE_INCLUDE_PATH} ${CMAKE_PREFIX_PATH}/include $ENV{OpenGM_ROOT}/include ENV CPLUS_INCLUDE_PATH)

# handle the QUIETLY and REQUIRED arguments and set Opengm_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenGM DEFAULT_MSG OpenGM_INCLUDE_DIR)

MARK_AS_ADVANCED( OpenGM_INCLUDE_DIR )
