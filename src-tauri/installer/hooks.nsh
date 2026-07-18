; S68e: the WebView2 profile (localStorage) and the logs moved INTO the install dir
; (webview\ / logs\) for full portability. The uninstaller's "Delete the application
; data" checkbox used to cover them via the identifier-dir wipe — keep that privacy
; parity by removing the merged locations too when (and only when) the box is checked.
; $DeleteAppDataCheckboxState is the template's own global (installer.nsi un.Confirm*).
!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
    RmDir /r "$INSTDIR\webview"
    RmDir /r "$INSTDIR\logs"
    RMDir "$INSTDIR"
  ${EndIf}
!macroend
