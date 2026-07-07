package com.familylens

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

/**
 * Package que registra o FaceRecognitionModule no React Native.
 *
 * Todo módulo nativo precisa de um "Package" correspondente —
 * é ele que o React Native usa para descobrir quais módulos estão disponíveis.
 */
class FaceRecognitionPackage : ReactPackage {

    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        return listOf(FaceRecognitionModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}
