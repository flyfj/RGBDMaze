//-----------------------------------------------------------------------------
// <copyright file="KinectHelper.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------------

#pragma once

#include "windows.h"
#include <NuiApi.h>
#include <stdlib.h>
#include <algorithm>
#include <iterator>

namespace Microsoft {
    namespace KinectBridge {
        template <typename Image>
        class KinectHelper
        {
            // Constants:
            // Default resolutions
            static const NUI_IMAGE_RESOLUTION COLOR_DEFAULT_RESOLUTION = NUI_IMAGE_RESOLUTION_640x480;
            static const NUI_IMAGE_RESOLUTION DEPTH_DEFAULT_RESOLUTION = NUI_IMAGE_RESOLUTION_320x240;

        public:
            // Functions:
            /// <summary>
            /// Constructor
            /// </summary>
            KinectHelper();

            /// <summary>
            /// Destructor
            /// </summary>
            virtual ~KinectHelper();

            /// <summary>
            /// Sets whether to use color, depth, skeleton, and player index
            /// </summary>
            /// <param name="useColor">whether to use color</param>
            /// <param name="useDepth">whether to use depth</param>
            /// <param name="useSkeleton">whether to use skeleton tracking</param>
            /// <param name="usePlayerIndex">whether to use player indices if depth is also enabled</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT SetNuiInitFlags(bool useColor, bool useDepth, bool useSkeleton, bool usePlayerIndex = true);

            /// <summary>
            /// Sets the color stream resolution
            /// </summary>
            /// <param name="res">resolution to use</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT SetColorFrameResolution(NUI_IMAGE_RESOLUTION resolution);

            /// <summary>
            /// Sets the depth stream resolution
            /// </summary>
            /// <param name="res">resolution to use</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT SetDepthFrameResolution(NUI_IMAGE_RESOLUTION resolution);

            /// <summary>
            /// Sets or clears the specified depth stream flag
            /// </summary>
            /// <param name="flag">flag to set or clear</param>
            /// <param name="value">true to set, false to clear</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT SetDepthStreamFlag(DWORD flag, bool value);

            /// <summary>
            /// Sets or clears the specified skeleton tracking flag
            /// </summary>
            /// <param name="flag">flag to set or clear</param>
            /// <param name="value">true to set, false to clear</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT SetSkeletonTrackingFlag(DWORD flag, bool value);

            /// <summary>
            /// Initializes the Kinect with the given sensor. The depth stream and color stream,
            /// if they are opened, will be set to the default resolutions defined in COLOR_DEFAULT_RESOLUTION
            /// and DEPTH_DEFAULT_RESOLUTION.
            /// </summary>
            /// <param name="pNuiSensor">sensor to initialize</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT Initialize(INuiSensor* pNuiSensor);

            /// <summary>
            /// Uninitializes the Kinect
            /// </summary>
            void UnInitialize();

            /// <summary>
            /// Returns whether or not the Kinect has been initialized
            /// </summary>
            /// <returns>true if the Kinect is initialized, false otherwise</returns>
            bool IsInitialized() const;

            /// <summary>
            /// Returns the device connection id of the Kinect sensor that KinectHelper is associated with
            /// </summary>
            /// <returns>device connection id of Kinect sensor</param>
            BSTR GetKinectDeviceConnectionId() const;

            /// <summary>
            /// Updates the internal color image
            /// </summary>
            /// <param name="waitMillis">number of milliseconds to wait</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT UpdateColorFrame(DWORD waitMillis = 0);

            /// <summary>
            /// Updates the internal depth image
            /// </summary>
            /// <param name="waitMillis">number of milliseconds to wait</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT UpdateDepthFrame(DWORD waitMillis = 0);

            /// <summary>
            /// Updates the internal skeleton frame
            /// </summary>
            /// <param name="waitMillis">number of milliseconds to wait</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT UpdateSkeletonFrame(DWORD waitMillis = 0);

            /// <summary>
            /// Gets the color stream resolution
            /// </summary>
            /// <param name="width">pointer to store width in</param>
            /// <param name="height">pointer to store depth in</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetColorFrameSize(DWORD* width, DWORD* height) const;

            /// <summary>
            /// Gets the depth stream resolution
            /// </summary>
            /// <param name="width">pointer to store width in</param>
            /// <param name="height">pointer to store depth in</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetDepthFrameSize(DWORD* width, DWORD* height) const;

            /// <summary>
            /// Gets the color frame event handle
            /// </summary>
            /// <param name="phColorEvent">pointer in which to return the handle</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetColorHandle(HANDLE* phColorEvent) const;

            /// <summary>
            /// Gets the depth frame event handle
            /// </summary>
            /// <param name="phDepthEvent">pointer in which to return the handle</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetDepthHandle(HANDLE* phDepthEvent) const;

            /// <summary>
            /// Gets the skeleton frame event handle
            /// </summary>
            /// <param name="phSkeletonEvent">pointer in which to return the handle</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetSkeletonHandle(HANDLE* phSkeletonEvent) const;

            /// <summary>
            /// Gets the color image
            /// </summary>
            /// <param name="pColorImage">pointer in which to return the image</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetColorImage(Image* pColorImage) const;

            /// <summary>
            /// Gets the depth image
            /// </summary>
            /// <param name="pDepthImage">pointer in which to return the image</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetDepthImage(Image* pDepthImage) const;

            /// <summary>
            /// Gets the skeleton frame
            /// </summary>
            /// <param name="pSkeletonFrame">pointer in which to return the skeleton frame</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetSkeletonFrame(NUI_SKELETON_FRAME* pSkeletonFrame) const;

            /// <summary>
            /// Gets the depth image in ARGB
            /// </summary>
            /// <param name="pDepthArgbImage">pointer in which to return the image</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT GetDepthImageAsArgb(Image* pDepthArgbImage) const;

        protected:
            // Functions:
            /// <summary>
            /// Converts from Kinect color frame data into Image frame data
            /// </summary>
            /// <param name="pImage">pointer in which to return the image data</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            virtual HRESULT GetColorData(Image* pImage) const = 0;

            /// <summary>
            /// Converts from Kinect depth frame data into Image frame data
            /// </summary>
            /// <param name="pImage">pointer in which to return the image data</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            virtual HRESULT GetDepthData(Image* pImage) const = 0;

            /// <summary>
            /// Converts from Kinect depth frame data into ARGB Image frame data
            /// </summary>
            /// <param name="pImage">pointer in which to return the image data</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            virtual HRESULT GetDepthDataAsArgb(Image* pImage) const = 0;

            /// <summary>
            /// Verify image is of the given resolution
            /// </summary>
            /// <param name="pImage">pointer to image to verify</param>
            /// <param name="resolution">resolution of image</param>
            /// <returns>S_OK if image matches given width and height, an error code otherwise</returns>
            virtual HRESULT VerifySize(const Image* pImage, NUI_IMAGE_RESOLUTION resolution) const = 0;

            /// <summary>
            /// Convert a 13-bit depth value into a set of RGB values
            /// </summary>
            /// <param name="depth">depth value to convert</param>
            /// <param name="pRedPixel">value of red pixel</param>
            /// <param name="pGreenPixel">value of green pixel</param>
            /// <param name="pBluePixel">value of blue pixel</param>
            /// <returns>S_OK if successful, an error code otherwise</returns>
            HRESULT DepthShortToRgb(USHORT depth, UINT8* pRedPixel, UINT8* pGreenPixel, UINT8* pBluePixel) const;

            // Image stream data
            BYTE* m_pColorBuffer;
            INT m_colorBufferSize;
            INT m_colorBufferPitch;
            BYTE* m_pDepthBuffer;
            INT m_depthBufferSize;
            INT m_depthBufferPitch;

            // Image stream resolution information
            NUI_IMAGE_RESOLUTION m_colorResolution;
            NUI_IMAGE_RESOLUTION m_depthResolution;

        private:
            // Functions:
            // Variables:
            // Image stream handles
            HANDLE m_hColorStreamHandle;
            HANDLE m_hDepthStreamHandle;

            // Frame event handles
            // These are handles to events created using the CreateEvent Win32 API
            HANDLE m_hNextColorFrameEvent;
            HANDLE m_hNextDepthFrameEvent;
            HANDLE m_hNextSkeletonFrameEvent;

            // Stream usage settings
            bool m_isUsingColor;
            bool m_isUsingDepth;
            bool m_isUsingSkeleton;
            bool m_isUsingPlayerIndex;

            // Kinect initialization flags
            DWORD m_nuiInitFlags;

            // Image stream initialization flags
            DWORD m_depthFlags;
            DWORD m_skeletonFlags;

            // Internal skeleton frame
            NUI_SKELETON_FRAME m_skeletonFrame;

            // Pointer to Kinect sensor
            INuiSensor* m_pNuiSensor;

        };


		//////////////////////////////////////////////////////////////////////////
		// implementation
		//////////////////////////////////////////////////////////////////////////

        /// <summary>
        /// Constructor
        /// </summary>
        template <typename Image>
        KinectHelper<Image>::KinectHelper() :
            m_hColorStreamHandle(NULL),
            m_hDepthStreamHandle(NULL),
            m_hNextColorFrameEvent(NULL),
            m_hNextDepthFrameEvent(NULL),
            m_hNextSkeletonFrameEvent(NULL),
            m_depthFlags(0),
            m_skeletonFlags(NUI_SKELETON_TRACKING_FLAG_ENABLE_IN_NEAR_RANGE),
            m_pNuiSensor(NULL),
            m_pColorBuffer(NULL),
            m_colorBufferSize(0),
            m_colorBufferPitch(0),
            m_pDepthBuffer(NULL),
            m_depthBufferSize(0),
            m_depthBufferPitch(0),
            m_colorResolution(COLOR_DEFAULT_RESOLUTION),
            m_depthResolution(DEPTH_DEFAULT_RESOLUTION)
        {
            // Default to all streams enabled
            SetNuiInitFlags(true, true, false);
        }

        /// <summary>
        /// Destructor
        /// </summary>
        template <typename Image>
        KinectHelper<Image>::~KinectHelper()
        {
            UnInitialize();
        }

        /// <summary>
        /// Sets whether to use color, depth, skeleton, and player index
        /// </summary>
        /// <param name="useColor">whether to use color</param>
        /// <param name="useDepth">whether to use depth</param>
        /// <param name="useSkeleton">whether to use skeleton tracking</param>
        /// <param name="usePlayerIndex">whether to use player indices if depth is also enabled</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::SetNuiInitFlags(bool useColor, bool useDepth, bool useSkeleton, bool usePlayerIndex /* = true */)
        {
            // Fail if Kinect is already initialized
            if (m_pNuiSensor) 
            {
                return E_NUI_ALREADY_INITIALIZED;
            }

            // Fail if no streams are used
            if (!(useColor || useDepth || useSkeleton)) 
            {
                return E_INVALIDARG;
            }

            // Update usage settings variables
            m_isUsingColor = useColor;
            m_isUsingDepth = useDepth;
            m_isUsingSkeleton = useSkeleton;
            m_isUsingPlayerIndex = usePlayerIndex;

            // Update flags and event count based on selected options
            m_nuiInitFlags = 0;

            if (useColor) 
            {
                m_nuiInitFlags |= NUI_INITIALIZE_FLAG_USES_COLOR;
            }

            if (useDepth) 
            {
                if (usePlayerIndex) 
                {
                    m_nuiInitFlags |= NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX;
                } 
                else 
                {
                    m_nuiInitFlags |= NUI_INITIALIZE_FLAG_USES_DEPTH;
                }
            }

            if (useSkeleton) 
            {
                m_nuiInitFlags |= NUI_INITIALIZE_FLAG_USES_SKELETON;
            }

            return S_OK;
        }

        /// <summary>
        /// Sets the color stream resolution
        /// </summary>
        /// <param name="res">resolution to use</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::SetColorFrameResolution(NUI_IMAGE_RESOLUTION resolution)
        {
            DWORD width, height;
            NuiImageResolutionToSize(resolution, width, height);

            // Do nothing if the new resolution is the same
            if (resolution == m_colorResolution)
            {
                return S_OK;
            }

            // Fail if resolution is invalid
            if (resolution != NUI_IMAGE_RESOLUTION_1280x960 && resolution != NUI_IMAGE_RESOLUTION_640x480) 
            {
                return E_INVALIDARG;
            }

            // Update color resolution variable
            m_colorResolution = resolution;

            HRESULT hr = S_OK;

            // If color stream is already opened, update its resolution
            if (m_pNuiSensor)
            {
                hr = m_pNuiSensor->NuiImageStreamOpen(
                    NUI_IMAGE_TYPE_COLOR,
                    resolution,
                    0,
                    2,
                    m_hNextColorFrameEvent,
                    &m_hColorStreamHandle);
            }

            return hr;
        }

        /// <summary>
        /// Sets the depth stream resolution
        /// </summary>
        /// <param name="res">resolution to use</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::SetDepthFrameResolution(NUI_IMAGE_RESOLUTION resolution)
        {
            DWORD width, height;
            NuiImageResolutionToSize(resolution, width, height);

            // Do nothing if the new resolution is the same
            if (resolution == m_depthResolution)
            {
                return S_OK;
            }

            // Fail if resolution is invalid
            if (resolution != NUI_IMAGE_RESOLUTION_640x480 && resolution != NUI_IMAGE_RESOLUTION_320x240 && resolution != NUI_IMAGE_RESOLUTION_80x60) 
            {
                return E_INVALIDARG;
            }

            // Update depth resolution variable
            m_depthResolution = resolution;

            HRESULT hr = S_OK;

            // If depth stream is already open, update its resolution
            if (m_pNuiSensor)
            {
                hr = m_pNuiSensor->NuiImageStreamOpen(
                    m_isUsingPlayerIndex ? NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX : NUI_IMAGE_TYPE_DEPTH,
                    resolution,
                    m_depthFlags,
                    2,
                    m_hNextDepthFrameEvent,
                    &m_hDepthStreamHandle);
            }

            return hr;
        }

        /// <summary>
        /// Sets or clears the specified depth stream flag
        /// </summary>
        /// <param name="flag">flag to set or clear</param>
        /// <param name="value">true to set, false to clear</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::SetDepthStreamFlag(DWORD flag, bool value)
        {
            // Fail if depth stream is not enabled
            if (!m_isUsingDepth) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Get value for new flags
            DWORD newFlags = m_depthFlags;
            if (value) 
            {
                newFlags |= flag;
            } 
            else 
            {
                newFlags &= ~flag;
            }

            // Apply new flags to depth stream
            if (newFlags != m_depthFlags) 
            {
                if (m_pNuiSensor)
                {
                    HRESULT hr = m_pNuiSensor->NuiImageStreamSetImageFrameFlags(m_hDepthStreamHandle, newFlags);

                    if (FAILED(hr))
                    {
                        return hr;
                    }
                }

                m_depthFlags = newFlags;
            }

            return S_OK;
        }

        /// <summary>
        /// Sets or clears the specified skeleton tracking flag
        /// </summary>
        /// <param name="flag">flag to set or clear</param>
        /// <param name="value">true to set, false to clear</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::SetSkeletonTrackingFlag(DWORD flag, bool value)
        {
            // Fail if skeleton tracking is not enabled
            if (!m_isUsingSkeleton) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Get value for new flags
            DWORD newFlags = m_skeletonFlags;
            if (value) 
            {
                newFlags |= flag;
            } 
            else 
            {
                newFlags &= ~flag;
            }

            // Apply new flags to skeleton tracking
            if (newFlags != m_skeletonFlags) 
            {
                if (m_pNuiSensor)
                {
                    HRESULT hr = m_pNuiSensor->NuiSkeletonTrackingEnable(m_hNextSkeletonFrameEvent, newFlags);

                    if (FAILED(hr))
                    {
                        return hr;
                    }
                }

                m_skeletonFlags = newFlags;
            }

            return S_OK;
        }

        /// <summary>
        /// Initializes the Kinect with the given sensor. The depth stream and color stream,
        /// if they are opened, will be set to the default resolutions defined in COLOR_DEFAULT_RESOLUTION
        //  and DEPTH_DEFAULT_RESOLUTION.
        /// </summary>
        /// <param name="pNuiSensor">sensor to initialize</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::Initialize(INuiSensor* pNuiSensor)
        {
            HRESULT hr;

            // If there is no sensor, initialize one
            if (!m_pNuiSensor)
            {
                if (!pNuiSensor)
                {
                    return E_POINTER;
                }
                m_pNuiSensor = pNuiSensor;

                hr = m_pNuiSensor->NuiInitialize(m_nuiInitFlags);
                if (FAILED(hr))
                {
                    return hr;
                }

            }

            // Create events based on usage settings
            if (m_isUsingColor) 
            {
                m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
            } 
            else 
            {
                m_hNextColorFrameEvent = INVALID_HANDLE_VALUE;
            }

            if (m_isUsingDepth) 
            {
                m_hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
            } 
            else 
            {
                m_hNextDepthFrameEvent = INVALID_HANDLE_VALUE;
            }

            if (m_isUsingSkeleton) 
            {
                m_hNextSkeletonFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
            } 
            else 
            {
                m_hNextSkeletonFrameEvent = INVALID_HANDLE_VALUE;
            }

            // Open image stream
            if (m_isUsingColor) 
            {
                hr = m_pNuiSensor->NuiImageStreamOpen(
                    NUI_IMAGE_TYPE_COLOR,
                    m_colorResolution,
                    0,
                    2,
                    m_hNextColorFrameEvent,
                    &m_hColorStreamHandle);
                if (FAILED(hr))
                {
                    return hr;
                }
            }

            // Open depth stream
            if (m_isUsingDepth) 
            {
                hr = m_pNuiSensor->NuiImageStreamOpen(
                    m_isUsingPlayerIndex ? NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX : NUI_IMAGE_TYPE_DEPTH,
                    m_depthResolution,
                    m_depthFlags,
                    2,
                    m_hNextDepthFrameEvent,
                    &m_hDepthStreamHandle);
                if (FAILED(hr))
                {
                    return hr;
                }
            }

            // Enable skeleton tracking
            if (m_isUsingSkeleton)
            {
                hr = m_pNuiSensor->NuiSkeletonTrackingEnable(m_hNextSkeletonFrameEvent, m_skeletonFlags);
                if (FAILED(hr))
                {
                    return hr;
                }
            }

            return hr;
        }

        /// <summary>
        /// Uninitializes the Kinect
        /// </summary>
        template <typename Image>
        void KinectHelper<Image>::UnInitialize()
        {
            // Close Kinect
            if (m_pNuiSensor)
            {
                m_pNuiSensor->NuiShutdown();
                m_pNuiSensor = NULL;
            }

            // Close handles for created events
            if (m_hNextColorFrameEvent && (m_hNextColorFrameEvent != INVALID_HANDLE_VALUE))
            {
                CloseHandle(m_hNextColorFrameEvent);
                m_hNextColorFrameEvent = NULL;
            }

            if (m_hNextDepthFrameEvent && (m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE))
            {
                CloseHandle(m_hNextDepthFrameEvent);
                m_hNextDepthFrameEvent = NULL;
            }

            if (m_hNextSkeletonFrameEvent && (m_hNextSkeletonFrameEvent != INVALID_HANDLE_VALUE))
            {
                CloseHandle(m_hNextSkeletonFrameEvent);
                m_hNextSkeletonFrameEvent = NULL;
            }
        }

        /// <summary>
        /// Returns whether or not the Kinect has been initialized
        /// </summary>
        /// <returns>true if the Kinect is initialized, false otherwise</returns>
        template <typename Image>
        bool KinectHelper<Image>::IsInitialized() const
        {
            return m_pNuiSensor != NULL;
        }

        /// <summary>
        /// Returns the device connection id of the Kinect sensor that KinectHelper is associated with
        /// </summary>
        /// <returns>device connection id of Kinect sensor</param>
        template <typename Image>
        BSTR KinectHelper<Image>::GetKinectDeviceConnectionId() const
        {
            return m_pNuiSensor->NuiDeviceConnectionId();
        }

        /// <summary>
        /// Updates the internal color image
        /// </summary>
        /// <param name="waitMillis">number of milliseconds to wait</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::UpdateColorFrame(DWORD waitMillis /* = 0 */)
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor)
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if color stream is not enabled
            if (!m_isUsingColor)
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Get next image stream frame
            NUI_IMAGE_FRAME imageFrame;

            HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(
                m_hColorStreamHandle,
                waitMillis,
                &imageFrame);
            if (FAILED(hr))
            {
                return hr;
            }

            // Lock frame texture to allow for copy
            INuiFrameTexture* pTexture = imageFrame.pFrameTexture;
            NUI_LOCKED_RECT lockedRect;
            pTexture->LockRect(0, &lockedRect, NULL, 0);

            // Check if image is valid
            if (lockedRect.Pitch != 0)
            {
                // Copy image information into buffer so it doesn't get overwritten later
                BYTE* pBuffer = lockedRect.pBits;
                INT size =  lockedRect.size;
                INT pitch = lockedRect.Pitch;

                // Only reallocate memory if the buffer size has changed
                if (size != m_colorBufferSize)
                {
                    delete [] m_pColorBuffer;
                    m_pColorBuffer = new BYTE[size];
                    m_colorBufferSize = size;
                }
                memcpy_s(m_pColorBuffer, size, pBuffer, size);


                m_colorBufferPitch = pitch;
            }

            // Unlock texture
            pTexture->UnlockRect(0);

            // Release image stream frame
            hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_hColorStreamHandle, &imageFrame);

            return hr;
        }

        /// <summary>
        /// Updates the internal depth image
        /// </summary>
        /// <param name="waitMillis">number of milliseconds to wait</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::UpdateDepthFrame(DWORD waitMillis /* = 0 */)
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor)
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if depth stream is not enabled
            if (!m_isUsingDepth)
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Get next image stream frame
            NUI_IMAGE_FRAME imageFrame;

            HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(
                m_hDepthStreamHandle,
                waitMillis,
                &imageFrame);
            if (FAILED(hr))
            {
                return hr;
            }

            // Lock frame texture to allow for copy
            INuiFrameTexture* pTexture = imageFrame.pFrameTexture;
            NUI_LOCKED_RECT lockedRect;
            pTexture->LockRect(0, &lockedRect, NULL, 0);

            // Check if image is valid
            if (lockedRect.Pitch != 0)
            {
                // Copy image information into buffer
                BYTE* pBuffer = lockedRect.pBits;
                INT size =  lockedRect.size;
                INT pitch = lockedRect.Pitch;

                // Only reallocate memory if the buffer size has changed
                if (size != m_depthBufferSize)
                {
                    delete [] m_pDepthBuffer;
                    m_pDepthBuffer = new BYTE[size];
                    m_depthBufferSize = size;
                }
                memcpy_s(m_pDepthBuffer, size, pBuffer, size);

                m_depthBufferPitch = pitch;
            }

            // Unlock texture
            pTexture->UnlockRect(0);

            // Release image stream frame
            hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_hDepthStreamHandle, &imageFrame);

            return hr;
        }

        /// <summary>
        /// Updates the internal skeleton frame
        /// </summary>
        /// <param name="waitMillis">number of milliseconds to wait</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::UpdateSkeletonFrame(DWORD waitMillis /* = 0 */)
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor)
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if skeleton tracking is not enabled
            if (!m_isUsingSkeleton)
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Get next skeleton frame
            HRESULT hr = m_pNuiSensor->NuiSkeletonGetNextFrame(waitMillis, &m_skeletonFrame);
            if (FAILED(hr))
            {
                return hr;
            }

            // Smooth skeletons
            hr = m_pNuiSensor->NuiTransformSmooth(&m_skeletonFrame,NULL);

            return hr;
        }

        /// <summary>
        /// Gets the color stream resolution
        /// </summary>
        /// <param name="width">pointer to store width in</param>
        /// <param name="height">pointer to store depth in</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetColorFrameSize(DWORD* width, DWORD* height) const
        {
            // Fail if pointer is invalid
            if (!width || !height) 
            {
                return E_POINTER;
            }

            NuiImageResolutionToSize(m_colorResolution, *width, *height);

            return S_OK;
        }

        /// <summary>
        /// Gets the depth stream resolution
        /// </summary>
        /// <param name="width">pointer to store width in</param>
        /// <param name="height">pointer to store depth in</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetDepthFrameSize(DWORD* width, DWORD* height) const
        {
            // Fail if pointer is invalid
            if (!width || !height) 
            {
                return E_POINTER;
            }

            NuiImageResolutionToSize(m_depthResolution, *width, *height);

            return S_OK;
        }

        /// <summary>
        /// Gets the color frame event handle
        /// </summary>
        /// <param name="phColorEvent">pointer in which to return the handle</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetColorHandle(HANDLE* phColorEvent) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if color stream is not enabled
            if (!m_isUsingColor) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!phColorEvent) 
            {
                return E_POINTER;
            }

            *phColorEvent = m_hNextColorFrameEvent;

            return S_OK;
        }

        /// <summary>
        /// Gets the depth frame event handle
        /// </summary>
        /// <param name="phDepthEvent">pointer in which to return the handle</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetDepthHandle(HANDLE* phDepthEvent) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if depth stream is not enabled
            if (!m_isUsingDepth) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!phDepthEvent) 
            {
                return E_POINTER;
            }

            *phDepthEvent = m_hNextDepthFrameEvent;

            return S_OK;
        }

        /// <summary>
        /// Gets the skeleton frame event handle
        /// </summary>
        /// <param name="phSkeletonEvent">pointer in which to return the handle</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetSkeletonHandle(HANDLE* phSkeletonEvent) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if skeleton tracking is not enabled
            if (!m_isUsingSkeleton) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!phSkeletonEvent) 
            {
                return E_POINTER;
            }

            *phSkeletonEvent = m_hNextSkeletonFrameEvent;

            return S_OK;
        }

        /// <summary>
        /// Gets the color image
        /// </summary>
        /// <param name="pColorImage">pointer in which to return the image</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetColorImage(Image* pColorImage) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if color stream is not enabled
            if (!m_isUsingColor) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!pColorImage) 
            {
                return E_POINTER;
            }

            // Fail if pColorImage is not the correct size
            HRESULT hr = VerifySize(pColorImage, m_colorResolution);
            if (FAILED(hr))
            {
                return hr;
            }

            hr = GetColorData(pColorImage);

            return hr;
        }

        /// <summary>
        /// Gets the depth image
        /// </summary>
        /// <param name="pDepthImage">pointer in which to return the image</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetDepthImage(Image* pDepthImage) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if depth stream is not enabled
            if (!m_isUsingDepth) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!pDepthImage) 
            {
                return E_POINTER;
            }

            // Fail if pDepthImage is not the correct size
            HRESULT hr = VerifySize(pDepthImage, m_depthWidth, m_depthHeight);
            if (FAILED(hr))
            {
                return hr;
            }

            hr = GetDepthData(pDepthImage);

            return hr;
        }

        /// <summary>
        /// Gets the skeleton frame
        /// </summary>
        /// <param name="pSkeletonFrame">pointer in which to return the skeleton frame</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetSkeletonFrame(NUI_SKELETON_FRAME* pSkeletonFrame) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if skeleton tracking is not enabled
            if (!m_isUsingSkeleton) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!pSkeletonFrame) 
            {
                return E_POINTER;
            }

            *pSkeletonFrame = m_skeletonFrame;

            return S_OK;
        }

        /// <summary>
        /// Gets the depth image in ARGB
        /// </summary>
        /// <param name="pDepthArgbImage">pointer in which to return the image</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::GetDepthImageAsArgb(Image* pDepthArgbImage) const
        {
            // Fail if Kinect is not initialized
            if (!m_pNuiSensor) 
            {
                return E_NUI_DEVICE_NOT_READY;
            }

            // Fail if depth stream is not enabled
            if (!m_isUsingDepth) 
            {
                return E_NUI_STREAM_NOT_ENABLED;
            }

            // Fail if pointer is invalid
            if (!pDepthArgbImage) 
            {
                return E_POINTER;
            }

            // Fail if pDepthImage is not the correct size
            HRESULT hr = VerifySize(pDepthArgbImage, m_depthResolution);
            if (FAILED(hr))
            {
                return hr;
            }

            // Update output Mat to correct size
            hr = GetDepthDataAsArgb(pDepthArgbImage);

            return hr;
        }

        /// <summary>
        /// Convert a 13-bit depth value into a set of RGB values
        /// </summary>
        /// <param name="depth">depth value to convert</param>
        /// <param name="pRedPixel">value of red pixel</param>
        /// <param name="pGreenPixel">value of green pixel</param>
        /// <param name="pBluePixel">value of blue pixel</param>
        /// <returns>S_OK if successful, an error code otherwise</returns>
        template <typename Image>
        HRESULT KinectHelper<Image>::DepthShortToRgb(USHORT depth, UINT8* redPixel, UINT8* greenPixel, UINT8* bluePixel) const
        {
            SHORT realDepth = NuiDepthPixelToDepth(depth);
            USHORT playerIndex = NuiDepthPixelToPlayerIndex(depth);

            // Convert depth info into an intensity for display
            BYTE b = 255 - static_cast<BYTE>(256 * realDepth / 0x0fff);

            // Color the output based on the player index
            switch(playerIndex)
            {
            case 0:
                *redPixel = b / 2;
                *greenPixel = b / 2;
                *bluePixel = b / 2;
                break;

            case 1:
                *redPixel = b;
                *greenPixel = 0;
                *bluePixel = 0;
                break;

            case 2:
                *redPixel = 0;
                *greenPixel = b;
                *bluePixel = 0;
                break;

            case 3:
                *redPixel = b / 4;
                *greenPixel = b;
                *bluePixel = b;
                break;

            case 4:
                *redPixel = b;
                *greenPixel = b;
                *bluePixel = b / 4;
                break;

            case 5:
                *redPixel = b;
                *greenPixel = b / 4;
                *bluePixel = b;
                break;

            case 6:
                *redPixel = b / 2;
                *greenPixel = b / 2;
                *bluePixel = b;
                break;

            case 7:
                *redPixel = 255 - (b / 2);
                *greenPixel = 255 - (b / 2);
                *bluePixel = 255 - (b / 2);
                break;

            default:
                *redPixel = 0;
                *greenPixel = 0;
                *bluePixel = 0;
                break;
            }


            return S_OK;
        }
    }
}

