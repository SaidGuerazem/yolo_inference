find_package(PkgConfig REQUIRED)

# Find GStreamer core and base libraries (required for pipelines and buffers)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0) # Critical for injecting/pulling frames

# Create a consolidated modern GStreamer interface target
add_library(GStreamer::GStreamer INTERFACE IMPORTED GLOBAL)
set_target_properties(GStreamer::GStreamer PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GSTREAMER_INCLUDE_DIRS};${GST_BASE_INCLUDE_DIRS};${GST_APP_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${GSTREAMER_LIBRARIES};${GST_BASE_LIBRARIES};${GST_APP_LIBRARIES}"
    INTERFACE_LINK_DIRECTORIES "${GSTREAMER_LIBRARY_DIRS};${GST_BASE_LIBRARY_DIRS};${GST_APP_LIBRARY_DIRS}"
)