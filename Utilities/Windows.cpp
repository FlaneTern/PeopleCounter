#include "pch.h"
#include "Utilities.h"



//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//

#include <windows.h>      // For common windows data types and function headers
#define STRICT_TYPED_ITEMIDS
#include <shlobj.h>
#include <objbase.h>      // For COM headers
#include <shobjidl.h>     // for IFileDialogEvents and IFileDialogControlEvents
#include <shlwapi.h>
#include <knownfolders.h> // for KnownFolder APIs/datatypes/function headers
#include <propvarutil.h>  // for PROPVAR-related functions
#include <propkey.h>      // for the Property key APIs/datatypes
#include <propidl.h>      // for the Property System APIs
#include <strsafe.h>      // for StringCchPrintfW
#include <shtypes.h>      // for COMDLG_FILTERSPEC
#include <new>

#pragma comment(linker, "\"/manifestdependency:type='Win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

const COMDLG_FILTERSPEC c_rgSaveTypes[] =
{
    {L"PNG File (*.png)",       L"*.png"},
    {L"JPG/JPEG File (*.jpg; *.jpeg)",    L"*.jpg;*.jpeg"},
};

// Indices of file types
#define INDEX_PNG 1
#define INDEX_JPG 2

// Controls
#define CONTROL_GROUP           2000
#define CONTROL_RADIOBUTTONLIST 2
#define CONTROL_RADIOBUTTON1    1
#define CONTROL_RADIOBUTTON2    2       // It is OK for this to have the same ID as CONTROL_RADIOBUTTONLIST,
                                        // because it is a child control under CONTROL_RADIOBUTTONLIST

// IDs for the Task Dialog Buttons
#define IDC_BASICFILEOPEN                       100
#define IDC_ADDITEMSTOCUSTOMPLACES              101
#define IDC_ADDCUSTOMCONTROLS                   102
#define IDC_SETDEFAULTVALUESFORPROPERTIES       103
#define IDC_WRITEPROPERTIESUSINGHANDLERS        104
#define IDC_WRITEPROPERTIESWITHOUTUSINGHANDLERS 105

/* File Dialog Event Handler *****************************************************************************************************/

class CDialogEventHandler : public IFileDialogEvents,
    public IFileDialogControlEvents
{
public:
    // IUnknown methods
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
    {
        static const QITAB qit[] = {
            QITABENT(CDialogEventHandler, IFileDialogEvents),
            QITABENT(CDialogEventHandler, IFileDialogControlEvents),
            { 0 },
#pragma warning(suppress:4838)
        };
        return QISearch(this, qit, riid, ppv);
    }

    IFACEMETHODIMP_(ULONG) AddRef()
    {
        return InterlockedIncrement(&_cRef);
    }

    IFACEMETHODIMP_(ULONG) Release()
    {
        long cRef = InterlockedDecrement(&_cRef);
        if (!cRef)
            delete this;
        return cRef;
    }

    // IFileDialogEvents methods
    IFACEMETHODIMP OnFileOk(IFileDialog*) { return S_OK; };
    IFACEMETHODIMP OnFolderChange(IFileDialog*) { return S_OK; };
    IFACEMETHODIMP OnFolderChanging(IFileDialog*, IShellItem*) { return S_OK; };
    IFACEMETHODIMP OnHelp(IFileDialog*) { return S_OK; };
    IFACEMETHODIMP OnSelectionChange(IFileDialog*) { return S_OK; };
    IFACEMETHODIMP OnShareViolation(IFileDialog*, IShellItem*, FDE_SHAREVIOLATION_RESPONSE*) { return S_OK; };
    IFACEMETHODIMP OnTypeChange(IFileDialog* pfd) { return S_OK; };
    IFACEMETHODIMP OnOverwrite(IFileDialog*, IShellItem*, FDE_OVERWRITE_RESPONSE*) { return S_OK; };

    // IFileDialogControlEvents methods
    IFACEMETHODIMP OnItemSelected(IFileDialogCustomize* pfdc, DWORD dwIDCtl, DWORD dwIDItem) { return S_OK; };
    IFACEMETHODIMP OnButtonClicked(IFileDialogCustomize*, DWORD) { return S_OK; };
    IFACEMETHODIMP OnCheckButtonToggled(IFileDialogCustomize*, DWORD, BOOL) { return S_OK; };
    IFACEMETHODIMP OnControlActivating(IFileDialogCustomize*, DWORD) { return S_OK; };

    CDialogEventHandler() : _cRef(1) { };
private:
    ~CDialogEventHandler() { };
    long _cRef;
};

// Instance creation helper
static HRESULT CDialogEventHandler_CreateInstance(REFIID riid, void** ppv)
{
    *ppv = NULL;
    CDialogEventHandler* pDialogEventHandler = new (std::nothrow) CDialogEventHandler();
    HRESULT hr = pDialogEventHandler ? S_OK : E_OUTOFMEMORY;
    if (SUCCEEDED(hr))
    {
        hr = pDialogEventHandler->QueryInterface(riid, ppv);
        pDialogEventHandler->Release();
    }
    return hr;
}

static void Initialize()
{
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
}





namespace Utilities
{
	bool IsKeyPressed(KeyCode key)
	{
		if (GetKeyState((int)key) & 0x8000)
			return true;
		return false;
	}   

	std::string GetFile()
    {
        static bool isInitialized = false;

        if (!isInitialized)
        {
            Initialize();
            isInitialized = true;
        }

        std::wstringstream filePathStream;

        // CoCreate the File Open Dialog object.
        IFileDialog* pfd = NULL;
        HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog,
            NULL,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&pfd));
        std::cout << *(uint32_t*)&hr;
        if (SUCCEEDED(hr))
        {
            // Create an event handling object, and hook it up to the dialog.
            IFileDialogEvents* pfde = NULL;
            hr = CDialogEventHandler_CreateInstance(IID_PPV_ARGS(&pfde));
            if (SUCCEEDED(hr))
            {
                // Hook up the event handler.
                DWORD dwCookie;
                hr = pfd->Advise(pfde, &dwCookie);
                if (SUCCEEDED(hr))
                {
                    // Set the options on the dialog.
                    DWORD dwFlags;

                    // Before setting, always get the options first in order 
                    // not to override existing options.
                    hr = pfd->GetOptions(&dwFlags);
                    if (SUCCEEDED(hr))
                    {
                        // In this case, get shell items only for file system items.
                        hr = pfd->SetOptions(dwFlags | FOS_FORCEFILESYSTEM);
                        if (SUCCEEDED(hr))
                        {
                            // Set the file types to display only. 
                            // Notice that this is a 1-based array.
                            hr = pfd->SetFileTypes(ARRAYSIZE(c_rgSaveTypes), c_rgSaveTypes);
                            if (SUCCEEDED(hr))
                            {
                                // Set the selected file type index to Word Docs for this example.
                                hr = pfd->SetFileTypeIndex(INDEX_PNG);
                                if (SUCCEEDED(hr))
                                {
                                    // Set the default extension to be ".doc" file.
                                    hr = pfd->SetDefaultExtension(L"doc;docx");
                                    if (SUCCEEDED(hr))
                                    {
                                        // Show the dialog
                                        hr = pfd->Show(NULL);
                                        if (SUCCEEDED(hr))
                                        {
                                            // Obtain the result once the user clicks 
                                            // the 'Open' button.
                                            // The result is an IShellItem object.
                                            IShellItem* psiResult;
                                            hr = pfd->GetResult(&psiResult);
                                            if (SUCCEEDED(hr))
                                            {
                                                // We are just going to print out the 
                                                // name of the file for sample sake.
                                                PWSTR pszFilePath = NULL;
                                                hr = psiResult->GetDisplayName(SIGDN_FILESYSPATH,
                                                    &pszFilePath);
                                                if (SUCCEEDED(hr))
                                                {
                                                    CoTaskMemFree(pszFilePath);
                                                }
                                                psiResult->Release();
                                                filePathStream << pszFilePath;

                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Unhook the event handler.
                    pfd->Unadvise(dwCookie);
                }
                pfde->Release();
            }
            pfd->Release();
        }

        std::wstring wstr = filePathStream.str();
        std::string strTo;
        char* szTo = new char[wstr.length() + 1];
        szTo[wstr.size()] = '\0';
        WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, szTo, (int)wstr.length(), NULL, NULL);
        strTo = szTo;
        delete[] szTo;
        
        std::cout << strTo;
        return strTo;
    }
}