#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <urlmon.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <locale>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <cwctype>

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "urlmon.lib")
#pragma comment(lib, "Advapi32.lib")

namespace fs = std::filesystem;

namespace {

constexpr UINT WMU_STATUS = WM_APP + 1;
constexpr UINT WMU_LOG = WM_APP + 2;
constexpr UINT WMU_FAILED = WM_APP + 3;
constexpr UINT WMU_LAUNCHED = WM_APP + 4;
constexpr int kProgressMax = 8;
constexpr wchar_t kWindowClassName[] = L"project_python_runtime_launcher";

struct UiMessage {
    int step = 0;
    std::wstring text;
};

struct WindowState {
    HWND status_label = nullptr;
    HWND progress_bar = nullptr;
    HWND log_edit = nullptr;
    HWND close_button = nullptr;
    HWND detail_label = nullptr;
    bool can_close = false;
    fs::path log_file;
};

struct LauncherConfig {
    std::wstring python_series = L"3.12";
    std::wstring fallback_python_version = L"3.12.10";
    std::wstring python_windows_release_url = L"https://www.python.org/getit/windows/";
    std::wstring python_installer_url_template = L"https://www.python.org/ftp/python/{version}/python-{version}-amd64.exe";
    std::wstring python_embeddable_url_template = L"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip";
    std::wstring runtime_dir_name = L"runtime";
    std::wstring python_base_dir_name = L"python-base";
    std::wstring python_env_dir_name = L"python";
    std::wstring cache_dir_name = L"cache";
    std::wstring logs_dir_name = L"logs";
    std::wstring python_version_file_name = L"python-version.txt";
    std::wstring torch_target_file_name = L"torch-target.txt";
    std::wstring filtered_requirements_file_name = L"requirements.filtered.txt";
    std::wstring torch_cuda_index_url = L"https://download.pytorch.org/whl/cu129";
    std::wstring torch_xpu_index_url = L"https://download.pytorch.org/whl/xpu";
    std::wstring torch_rocm_index_url = L"https://download.pytorch.org/whl/rocm6.2.4";
};

struct ProjectLayout {
    fs::path launcher_dir;
    fs::path project_root;
    fs::path runtime_dir;
    fs::path cache_dir;
    fs::path python_layout_dir;
    fs::path logs_dir;
    fs::path log_file;
    fs::path python_base_dir;
    fs::path python_env_dir;
    fs::path python_base_exe;
    fs::path python_basew_exe;
    fs::path python_exe;
    fs::path pythonw_exe;
    fs::path python_version_file;
    fs::path torch_target_file;
    fs::path filtered_requirements_file;
    fs::path python_index_cache_file;
    fs::path python_installer_file;
    fs::path python_embeddable_file;
    fs::path python_installer_log_file;
    fs::path requirements_file;
    fs::path run_script;
    fs::path pip_cache_dir;
};

enum class AcceleratorKind {
    NvidiaCuda,
    IntelArcXpu,
    AmdRocmDetected,
    Cpu,
};

struct AcceleratorChoice {
    AcceleratorKind kind = AcceleratorKind::Cpu;
    std::wstring display_name = L"CPU";
    std::wstring install_tag = L"cpu";
    std::optional<std::wstring> index_url;
    std::wstring note;
};

struct WorkerContext {
    LauncherConfig config;
    ProjectLayout layout;
    HWND hwnd = nullptr;
    std::vector<std::wstring> extra_args;
};

std::wstring trim_copy(std::wstring value);
std::wstring lower_copy(std::wstring value);
std::string utf8_from_wide(const std::wstring& value);
std::wstring wide_from_utf8(const std::string& value);
std::runtime_error make_error(const std::wstring& message);
void fail_if(bool condition, const std::wstring& message);
std::wstring last_error_message(DWORD error);
std::wstring format_hresult_hex(HRESULT hr);
std::wstring current_timestamp_local();
void ensure_parent_directory(const fs::path& path);
void write_text_file_utf8(const fs::path& path, const std::string& content);
std::wstring read_text_file_utf8(const fs::path& path);
void append_log_line(const fs::path& path, const std::wstring& message);
void post_message(HWND hwnd, UINT message_id, int step, const std::wstring& text);
void post_status(HWND hwnd, int step, const std::wstring& text);
void post_log(HWND hwnd, const std::wstring& text);
void post_failed(HWND hwnd, const std::wstring& text);
void post_launched(HWND hwnd, const std::wstring& text);
fs::path executable_path();
std::wstring quote_argument(const std::wstring& argument);
std::wstring build_command_line(const fs::path& executable, const std::vector<std::wstring>& arguments);
HANDLE open_log_handle_append(const fs::path& path);
int run_process(
    const fs::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::optional<fs::path>& working_dir,
    const fs::path& log_file,
    bool hidden = true
);
void start_detached_process(
    const fs::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::optional<fs::path>& working_dir,
    bool hidden
);
std::optional<fs::path> find_executable_on_path(const std::wstring& executable_name);
std::wstring quote_powershell_literal(const std::wstring& value);
void download_file(const std::wstring& url, const fs::path& destination, const fs::path& log_file);
std::wstring replace_all(std::wstring text, std::wstring_view from, std::wstring_view to);
std::wstring read_remote_text(const std::wstring& url, const fs::path& cache_file, const fs::path& log_file);
std::vector<int> parse_version_numbers(const std::wstring& version);
bool version_less(const std::wstring& left, const std::wstring& right);
std::wstring resolve_latest_python_version(const LauncherConfig& config, const ProjectLayout& layout);
std::optional<std::wstring> read_registry_default_value(HKEY root, const std::wstring& subkey);
std::optional<std::wstring> read_registry_string_value(HKEY root, const std::wstring& subkey, const std::wstring& value_name);
std::optional<fs::path> find_registered_python_install_dir(const LauncherConfig& config);
bool paths_equivalent_loose(const fs::path& left, const fs::path& right);
void cleanup_local_python_registration(const LauncherConfig& config, const ProjectLayout& layout);
bool command_exists_on_path(const std::wstring& executable_name);
std::vector<std::wstring> enumerate_display_adapter_names();
bool contains_case_insensitive(const std::wstring& text, const std::wstring& needle);
std::optional<std::wstring> env_value(const std::wstring& name);
AcceleratorChoice detect_accelerator(const LauncherConfig& config, const ProjectLayout& layout);
std::wstring python_installer_url_for_version(const LauncherConfig& config, const std::wstring& version);
std::wstring python_embeddable_url_for_version(const LauncherConfig& config, const std::wstring& version);
fs::path windows_system_executable(const std::wstring& executable_name);
void extract_zip_archive(const fs::path& archive_file, const fs::path& destination_dir, const fs::path& log_file);
void download_python_layout(const LauncherConfig& config, const ProjectLayout& layout);
std::optional<std::wstring> python_package_cache_version(const fs::path& candidate_path, const LauncherConfig& config);
std::optional<fs::path> locate_python_runtime_msi(
    const LauncherConfig& config,
    const ProjectLayout& layout,
    const std::wstring& file_name
);
void extract_python_runtime_msi(const ProjectLayout& layout, const fs::path& msi_path);
void write_embeddable_runtime_pth(const LauncherConfig& config, const ProjectLayout& layout);
void set_child_environment(const ProjectLayout& layout);
bool python_supports_tkinter(const fs::path& python_exe, const fs::path& log_file);
bool python_supports_launcher_runtime(const fs::path& python_exe, const fs::path& log_file);
bool python_has_importable_torch_stack(const fs::path& python_exe, const fs::path& log_file);
void create_local_venv(const fs::path& base_python_exe, const ProjectLayout& layout);
void repair_base_python(const ProjectLayout& layout);
void install_python_from_msi_payloads(const LauncherConfig& config, const ProjectLayout& layout, const std::wstring& python_version);
void install_base_python(const LauncherConfig& config, const ProjectLayout& layout, const std::wstring& python_version);
void ensure_python_runtime(const LauncherConfig& config, const ProjectLayout& layout, const std::wstring& python_version);
void run_python(const ProjectLayout& layout, const std::vector<std::wstring>& arguments, const std::optional<fs::path>& cwd);
void ensure_packaging_tools(const ProjectLayout& layout);
std::wstring canonical_requirement_name(const std::wstring& requirement_line);
std::optional<fs::path> prepare_filtered_requirements(const ProjectLayout& layout);
std::vector<std::wstring> pip_install_arguments(
    const std::vector<std::wstring>& packages,
    const std::optional<std::wstring>& index_url
);
void uninstall_torch_packages(const ProjectLayout& layout);
void install_torch_stack(const ProjectLayout& layout, const AcceleratorChoice& accelerator);
void install_project_requirements(const ProjectLayout& layout);
void launch_run_script(const ProjectLayout& layout, const std::vector<std::wstring>& extra_args);
ProjectLayout resolve_project_layout(const LauncherConfig& config);
void bootstrap_worker(WorkerContext context);
void append_log_to_edit(HWND edit_control, const std::wstring& text);
LRESULT CALLBACK launcher_window_proc(HWND hwnd, UINT message, WPARAM w_param, LPARAM l_param);
HWND create_launcher_window(HINSTANCE instance, const fs::path& log_file);
std::vector<std::wstring> parse_extra_arguments();

std::wstring trim_copy(std::wstring value) {
    const auto is_space = [](wchar_t ch) {
        return ch == L' ' || ch == L'\t' || ch == L'\r' || ch == L'\n';
    };
    while (!value.empty() && is_space(value.front())) {
        value.erase(value.begin());
    }
    while (!value.empty() && is_space(value.back())) {
        value.pop_back();
    }
    return value;
}

std::wstring lower_copy(std::wstring value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](wchar_t ch) { return static_cast<wchar_t>(std::towlower(ch)); }
    );
    return value;
}

std::string utf8_from_wide(const std::wstring& value) {
    if (value.empty()) {
        return "";
    }
    const int length = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (length <= 0) {
        throw std::runtime_error("Failed to convert UTF-16 to UTF-8.");
    }
    std::string result(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, result.data(), length, nullptr, nullptr);
    if (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

std::wstring wide_from_utf8(const std::string& value) {
    if (value.empty()) {
        return L"";
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (length <= 0) {
        throw std::runtime_error("Failed to convert UTF-8 to UTF-16.");
    }
    std::wstring result(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, result.data(), length);
    if (!result.empty() && result.back() == L'\0') {
        result.pop_back();
    }
    return result;
}

std::runtime_error make_error(const std::wstring& message) {
    return std::runtime_error(utf8_from_wide(message));
}

void fail_if(bool condition, const std::wstring& message) {
    if (condition) {
        throw make_error(message);
    }
}

std::wstring last_error_message(DWORD error) {
    LPWSTR buffer = nullptr;
    const DWORD size = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&buffer),
        0,
        nullptr
    );
    std::wstring message = size && buffer ? buffer : L"Unknown Windows error.";
    if (buffer) {
        LocalFree(buffer);
    }
    return trim_copy(message);
}

std::wstring format_hresult_hex(HRESULT hr) {
    wchar_t buffer[11]{};
    _snwprintf_s(buffer, _countof(buffer), _TRUNCATE, L"%08lX", static_cast<unsigned long>(hr));
    return buffer;
}

std::wstring current_timestamp_local() {
    SYSTEMTIME st{};
    GetLocalTime(&st);
    wchar_t buffer[64]{};
    swprintf_s(
        buffer,
        L"%04u-%02u-%02u %02u:%02u:%02u",
        static_cast<unsigned>(st.wYear),
        static_cast<unsigned>(st.wMonth),
        static_cast<unsigned>(st.wDay),
        static_cast<unsigned>(st.wHour),
        static_cast<unsigned>(st.wMinute),
        static_cast<unsigned>(st.wSecond)
    );
    return buffer;
}

void ensure_parent_directory(const fs::path& path) {
    if (!path.empty()) {
        fs::create_directories(path.parent_path());
    }
}

void write_text_file_utf8(const fs::path& path, const std::string& content) {
    ensure_parent_directory(path);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw make_error(L"Could not open " + path.wstring() + L" for writing.");
    }
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::wstring read_text_file_utf8(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return L"";
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return wide_from_utf8(buffer.str());
}

void append_log_line(const fs::path& path, const std::wstring& message) {
    ensure_parent_directory(path);
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    if (!stream) {
        return;
    }
    const std::string encoded = utf8_from_wide(L"[" + current_timestamp_local() + L"] " + message + L"\r\n");
    stream.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
}

void post_message(HWND hwnd, UINT message_id, int step, const std::wstring& text) {
    auto* payload = new UiMessage{step, text};
    PostMessageW(hwnd, message_id, 0, reinterpret_cast<LPARAM>(payload));
}

void post_status(HWND hwnd, int step, const std::wstring& text) {
    post_message(hwnd, WMU_STATUS, step, text);
}

void post_log(HWND hwnd, const std::wstring& text) {
    post_message(hwnd, WMU_LOG, 0, text);
}

void post_failed(HWND hwnd, const std::wstring& text) {
    post_message(hwnd, WMU_FAILED, kProgressMax, text);
}

void post_launched(HWND hwnd, const std::wstring& text) {
    post_message(hwnd, WMU_LAUNCHED, kProgressMax, text);
}

fs::path executable_path() {
    std::wstring buffer(MAX_PATH, L'\0');
    while (true) {
        const DWORD written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        fail_if(written == 0, L"Could not determine the launcher executable path: " + last_error_message(GetLastError()));
        if (written < buffer.size() - 1) {
            buffer.resize(written);
            return fs::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
}

std::wstring quote_argument(const std::wstring& argument) {
    if (argument.empty()) {
        return L"\"\"";
    }
    const bool needs_quotes = argument.find_first_of(L" \t\"") != std::wstring::npos;
    if (!needs_quotes) {
        return argument;
    }
    std::wstring quoted = L"\"";
    for (size_t index = 0; index < argument.size(); ++index) {
        const wchar_t ch = argument[index];
        if (ch == L'\\') {
            size_t slash_count = 0;
            while (index < argument.size() && argument[index] == L'\\') {
                ++slash_count;
                ++index;
            }
            if (index == argument.size()) {
                quoted.append(slash_count * 2, L'\\');
                break;
            }
            if (argument[index] == L'"') {
                quoted.append(slash_count * 2 + 1, L'\\');
                quoted += L'"';
            } else {
                quoted.append(slash_count, L'\\');
                quoted += argument[index];
            }
        } else if (ch == L'"') {
            quoted += L"\\\"";
        } else {
            quoted += ch;
        }
    }
    quoted += L"\"";
    return quoted;
}

std::wstring build_command_line(const fs::path& executable, const std::vector<std::wstring>& arguments) {
    std::wstring command_line = quote_argument(executable.wstring());
    for (const auto& argument : arguments) {
        command_line += L" ";
        command_line += quote_argument(argument);
    }
    return command_line;
}

HANDLE open_log_handle_append(const fs::path& path) {
    ensure_parent_directory(path);
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE handle = CreateFileW(
        path.c_str(),
        FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        &security,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    fail_if(handle == INVALID_HANDLE_VALUE, L"Could not open " + path.wstring() + L" for logging: " + last_error_message(GetLastError()));
    SetFilePointer(handle, 0, nullptr, FILE_END);
    return handle;
}

int run_process(
    const fs::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::optional<fs::path>& working_dir,
    const fs::path& log_file,
    bool hidden
) {
    fail_if(!fs::exists(executable), L"Executable not found: " + executable.wstring());

    std::wstring command_line = build_command_line(executable, arguments);
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    HANDLE log_handle = open_log_handle_append(log_file);
    HANDLE stdin_handle = GetStdHandle(STD_INPUT_HANDLE);
    if (stdin_handle == nullptr || stdin_handle == INVALID_HANDLE_VALUE) {
        stdin_handle = nullptr;
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = stdin_handle;
    startup.hStdOutput = log_handle;
    startup.hStdError = log_handle;
    if (hidden) {
        startup.dwFlags |= STARTF_USESHOWWINDOW;
        startup.wShowWindow = SW_HIDE;
    }

    PROCESS_INFORMATION process{};
    const DWORD flags = hidden ? CREATE_NO_WINDOW : 0;
    const std::wstring cwd = working_dir ? working_dir->wstring() : L"";
    const BOOL created = CreateProcessW(
        executable.c_str(),
        mutable_command.data(),
        nullptr,
        nullptr,
        TRUE,
        flags,
        nullptr,
        working_dir ? cwd.c_str() : nullptr,
        &startup,
        &process
    );
    CloseHandle(log_handle);

    fail_if(!created, L"Could not start " + executable.wstring() + L": " + last_error_message(GetLastError()));

    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 0;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}

void start_detached_process(
    const fs::path& executable,
    const std::vector<std::wstring>& arguments,
    const std::optional<fs::path>& working_dir,
    bool hidden
) {
    fail_if(!fs::exists(executable), L"Executable not found: " + executable.wstring());

    std::wstring command_line = build_command_line(executable, arguments);
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    if (hidden) {
        startup.dwFlags = STARTF_USESHOWWINDOW;
        startup.wShowWindow = SW_HIDE;
    }

    PROCESS_INFORMATION process{};
    const DWORD flags = hidden ? CREATE_NO_WINDOW : 0;
    const std::wstring cwd = working_dir ? working_dir->wstring() : L"";
    const BOOL created = CreateProcessW(
        executable.c_str(),
        mutable_command.data(),
        nullptr,
        nullptr,
        FALSE,
        flags,
        nullptr,
        working_dir ? cwd.c_str() : nullptr,
        &startup,
        &process
    );
    fail_if(!created, L"Could not launch " + executable.wstring() + L": " + last_error_message(GetLastError()));

    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
}

std::optional<fs::path> find_executable_on_path(const std::wstring& executable_name) {
    const DWORD required = SearchPathW(nullptr, executable_name.c_str(), nullptr, 0, nullptr, nullptr);
    if (required == 0) {
        return std::nullopt;
    }
    std::wstring buffer(required, L'\0');
    const DWORD written = SearchPathW(nullptr, executable_name.c_str(), nullptr, required, buffer.data(), nullptr);
    if (written == 0 || written >= required) {
        return std::nullopt;
    }
    buffer.resize(written);
    return fs::path(buffer);
}

std::wstring quote_powershell_literal(const std::wstring& value) {
    std::wstring quoted = L"'";
    for (const wchar_t ch : value) {
        if (ch == L'\'') {
            quoted += L"''";
        } else {
            quoted += ch;
        }
    }
    quoted += L"'";
    return quoted;
}

void download_file(const std::wstring& url, const fs::path& destination, const fs::path& log_file) {
    ensure_parent_directory(destination);
    const fs::path temp_destination = destination.wstring() + L".part";
    if (fs::exists(temp_destination)) {
        std::error_code ignore;
        fs::remove(temp_destination, ignore);
    }

    append_log_line(log_file, L"Downloading " + url + L" -> " + destination.wstring());

    const auto finalize_download = [&]() {
        fail_if(!fs::exists(temp_destination), L"Download finished, but the temporary file is missing: " + temp_destination.wstring());
        if (fs::exists(destination)) {
            std::error_code ignore;
            fs::remove(destination, ignore);
        }
        fs::rename(temp_destination, destination);
    };

    const HRESULT hr = URLDownloadToFileW(nullptr, url.c_str(), temp_destination.c_str(), 0, nullptr);
    if (SUCCEEDED(hr)) {
        finalize_download();
        append_log_line(log_file, L"Download completed with URLMON.");
        return;
    }

    append_log_line(log_file, L"URLMON download failed with HRESULT 0x" + format_hresult_hex(hr) + L". Trying fallback download methods.");
    if (fs::exists(temp_destination)) {
        std::error_code ignore;
        fs::remove(temp_destination, ignore);
    }

    if (const auto curl_path = find_executable_on_path(L"curl.exe")) {
        append_log_line(log_file, L"Trying curl fallback with " + curl_path->wstring());
        const int curl_exit = run_process(
            *curl_path,
            {
                L"-L",
                L"--fail",
                L"--retry", L"3",
                L"--retry-delay", L"2",
                L"--output", temp_destination.wstring(),
                url
            },
            std::nullopt,
            log_file,
            true
        );
        if (curl_exit == 0 && fs::exists(temp_destination)) {
            finalize_download();
            append_log_line(log_file, L"Download completed with curl.");
            return;
        }
        append_log_line(log_file, L"curl fallback failed with exit code " + wide_from_utf8(std::to_string(curl_exit)) + L".");
        if (fs::exists(temp_destination)) {
            std::error_code ignore;
            fs::remove(temp_destination, ignore);
        }
    } else {
        append_log_line(log_file, L"curl.exe was not found on PATH.");
    }

    if (const auto powershell_path = find_executable_on_path(L"powershell.exe")) {
        append_log_line(log_file, L"Trying PowerShell Invoke-WebRequest fallback with " + powershell_path->wstring());
        const std::wstring command =
            L"$ProgressPreference='SilentlyContinue'; "
            L"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
            L"Invoke-WebRequest -Uri " + quote_powershell_literal(url)
            + L" -OutFile " + quote_powershell_literal(temp_destination.wstring())
            + L" -UseBasicParsing";
        const int powershell_exit = run_process(
            *powershell_path,
            {
                L"-NoProfile",
                L"-ExecutionPolicy", L"Bypass",
                L"-Command", command
            },
            std::nullopt,
            log_file,
            true
        );
        if (powershell_exit == 0 && fs::exists(temp_destination)) {
            finalize_download();
            append_log_line(log_file, L"Download completed with PowerShell.");
            return;
        }
        append_log_line(log_file, L"PowerShell fallback failed with exit code " + wide_from_utf8(std::to_string(powershell_exit)) + L".");
        if (fs::exists(temp_destination)) {
            std::error_code ignore;
            fs::remove(temp_destination, ignore);
        }
    } else {
        append_log_line(log_file, L"powershell.exe was not found on PATH.");
    }

    throw make_error(
        L"Download failed for "
        + url
        + L". URLMON returned HRESULT 0x"
        + format_hresult_hex(hr)
        + L", and the fallback download methods did not succeed either."
    );
}

std::wstring replace_all(std::wstring text, std::wstring_view from, std::wstring_view to) {
    size_t position = 0;
    while ((position = text.find(from, position)) != std::wstring::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
    }
    return text;
}

std::wstring read_remote_text(const std::wstring& url, const fs::path& cache_file, const fs::path& log_file) {
    try {
        download_file(url, cache_file, log_file);
        return read_text_file_utf8(cache_file);
    } catch (...) {
        if (fs::exists(cache_file)) {
            append_log_line(log_file, L"Falling back to cached response at " + cache_file.wstring());
            return read_text_file_utf8(cache_file);
        }
        throw;
    }
}

std::vector<int> parse_version_numbers(const std::wstring& version) {
    std::vector<int> parts;
    std::wstring current;
    for (const wchar_t ch : version) {
        if (ch == L'.') {
            if (!current.empty()) {
                parts.push_back(std::stoi(current));
                current.clear();
            }
            continue;
        }
        if (!std::iswdigit(ch)) {
            break;
        }
        current += ch;
    }
    if (!current.empty()) {
        parts.push_back(std::stoi(current));
    }
    return parts;
}

bool version_less(const std::wstring& left, const std::wstring& right) {
    const auto left_parts = parse_version_numbers(left);
    const auto right_parts = parse_version_numbers(right);
    const size_t count = std::max(left_parts.size(), right_parts.size());
    for (size_t index = 0; index < count; ++index) {
        const int left_value = index < left_parts.size() ? left_parts[index] : 0;
        const int right_value = index < right_parts.size() ? right_parts[index] : 0;
        if (left_value < right_value) {
            return true;
        }
        if (left_value > right_value) {
            return false;
        }
    }
    return false;
}

std::wstring resolve_latest_python_version(const LauncherConfig& config, const ProjectLayout& layout) {
    append_log_line(
        layout.log_file,
        L"Resolving the latest Windows-installable Python "
        + config.python_series
        + L" release from "
        + config.python_windows_release_url
    );
    std::wstring latest = config.fallback_python_version;
    try {
        const std::wstring listing = read_remote_text(config.python_windows_release_url, layout.python_index_cache_file, layout.log_file);
        const std::wregex pattern(L"Python (" + config.python_series + L"(\\.\\d+))");
        for (std::wsregex_iterator it(listing.begin(), listing.end(), pattern), end; it != end; ++it) {
            const std::wstring candidate = (*it)[1].str();
            if (version_less(latest, candidate)) {
                latest = candidate;
            }
        }
    } catch (const std::exception& exc) {
        append_log_line(layout.log_file, L"Could not refresh the Windows release listing. Using fallback " + latest + L".");
        append_log_line(layout.log_file, wide_from_utf8(exc.what()));
    }
    append_log_line(layout.log_file, L"Selected Windows-installable Python runtime version " + latest);
    return latest;
}

std::optional<std::wstring> read_registry_default_value(HKEY root, const std::wstring& subkey) {
    HKEY key = nullptr;
    if (RegOpenKeyExW(root, subkey.c_str(), 0, KEY_READ, &key) != ERROR_SUCCESS) {
        return std::nullopt;
    }

    DWORD type = 0;
    DWORD size = 0;
    const LONG size_result = RegQueryValueExW(key, nullptr, nullptr, &type, nullptr, &size);
    if (size_result != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ) || size == 0) {
        RegCloseKey(key);
        return std::nullopt;
    }

    std::wstring buffer(size / sizeof(wchar_t), L'\0');
    const LONG read_result = RegQueryValueExW(
        key,
        nullptr,
        nullptr,
        &type,
        reinterpret_cast<LPBYTE>(buffer.data()),
        &size
    );
    RegCloseKey(key);
    if (read_result != ERROR_SUCCESS) {
        return std::nullopt;
    }
    if (!buffer.empty() && buffer.back() == L'\0') {
        buffer.pop_back();
    }
    return trim_copy(buffer);
}

std::optional<std::wstring> read_registry_string_value(HKEY root, const std::wstring& subkey, const std::wstring& value_name) {
    HKEY key = nullptr;
    if (RegOpenKeyExW(root, subkey.c_str(), 0, KEY_READ, &key) != ERROR_SUCCESS) {
        return std::nullopt;
    }

    DWORD type = 0;
    DWORD size = 0;
    const LONG size_result = RegQueryValueExW(key, value_name.c_str(), nullptr, &type, nullptr, &size);
    if (size_result != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ) || size == 0) {
        RegCloseKey(key);
        return std::nullopt;
    }

    std::wstring buffer(size / sizeof(wchar_t), L'\0');
    const LONG read_result = RegQueryValueExW(
        key,
        value_name.c_str(),
        nullptr,
        &type,
        reinterpret_cast<LPBYTE>(buffer.data()),
        &size
    );
    RegCloseKey(key);
    if (read_result != ERROR_SUCCESS) {
        return std::nullopt;
    }
    if (!buffer.empty() && buffer.back() == L'\0') {
        buffer.pop_back();
    }
    return trim_copy(buffer);
}

std::optional<fs::path> find_registered_python_install_dir(const LauncherConfig& config) {
    const std::wstring subkey = L"Software\\Python\\PythonCore\\" + config.python_series + L"\\InstallPath";
    if (const auto value = read_registry_default_value(HKEY_CURRENT_USER, subkey)) {
        return fs::path(*value);
    }
    if (const auto value = read_registry_default_value(HKEY_LOCAL_MACHINE, subkey)) {
        return fs::path(*value);
    }
    const std::wstring wow_subkey = L"Software\\WOW6432Node\\Python\\PythonCore\\" + config.python_series + L"\\InstallPath";
    if (const auto value = read_registry_default_value(HKEY_LOCAL_MACHINE, wow_subkey)) {
        return fs::path(*value);
    }
    return std::nullopt;
}

bool paths_equivalent_loose(const fs::path& left, const fs::path& right) {
    std::error_code exists_error;
    if (fs::exists(left, exists_error) && fs::exists(right, exists_error)) {
        std::error_code equivalent_error;
        if (fs::equivalent(left, right, equivalent_error) && !equivalent_error) {
            return true;
        }
    }
    auto normalize = [](fs::path value) {
        std::wstring text = lower_copy(value.lexically_normal().wstring());
        while (text.size() > 3 && (text.back() == L'\\' || text.back() == L'/')) {
            text.pop_back();
        }
        return text;
    };
    return normalize(left) == normalize(right);
}

void cleanup_local_python_registration(const LauncherConfig& config, const ProjectLayout& layout) {
    const std::wstring python_core_key = L"Software\\Python\\PythonCore\\" + config.python_series;
    const std::wstring install_path_key = python_core_key + L"\\InstallPath";
    const auto registered_install_dir = read_registry_default_value(HKEY_CURRENT_USER, install_path_key);
    const bool local_python_registered = registered_install_dir.has_value()
        && paths_equivalent_loose(fs::path(*registered_install_dir), layout.python_base_dir);

    if (!local_python_registered) {
        return;
    }

    append_log_line(
        layout.log_file,
        L"Removing Windows registration for the project-local Python runtime at " + layout.python_base_dir.wstring()
    );

    const std::wstring uninstall_root = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall";
    std::vector<std::wstring> uninstall_subkeys;
    std::vector<fs::path> bundle_cache_dirs;

    HKEY uninstall_key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, uninstall_root.c_str(), 0, KEY_READ, &uninstall_key) == ERROR_SUCCESS) {
        DWORD index = 0;
        wchar_t subkey_name[256];
        DWORD subkey_size = _countof(subkey_name);
        while (true) {
            subkey_size = _countof(subkey_name);
            const LONG result = RegEnumKeyExW(
                uninstall_key,
                index,
                subkey_name,
                &subkey_size,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );
            if (result == ERROR_NO_MORE_ITEMS) {
                break;
            }
            if (result == ERROR_SUCCESS) {
                const std::wstring child = uninstall_root + L"\\" + subkey_name;
                const auto provider_key = read_registry_string_value(HKEY_CURRENT_USER, child, L"BundleProviderKey");
                const auto display_name = read_registry_string_value(HKEY_CURRENT_USER, child, L"DisplayName");
                const auto publisher = read_registry_string_value(HKEY_CURRENT_USER, child, L"Publisher");
                if (provider_key.has_value()
                    && display_name.has_value()
                    && publisher.has_value()
                    && *provider_key == L"CPython-" + config.python_series
                    && lower_copy(*publisher) == lower_copy(L"Python Software Foundation")
                    && lower_copy(*display_name).rfind(lower_copy(L"Python " + config.python_series), 0) == 0) {
                    uninstall_subkeys.push_back(child);
                    if (const auto bundle_cache = read_registry_string_value(HKEY_CURRENT_USER, child, L"BundleCachePath")) {
                        bundle_cache_dirs.push_back(fs::path(*bundle_cache).parent_path());
                    }
                }
            }
            ++index;
        }
        RegCloseKey(uninstall_key);
    }

    std::error_code fs_error;
    for (const auto& subkey : uninstall_subkeys) {
        const LONG delete_result = RegDeleteTreeW(HKEY_CURRENT_USER, subkey.c_str());
        if (delete_result == ERROR_SUCCESS || delete_result == ERROR_FILE_NOT_FOUND) {
            append_log_line(layout.log_file, L"Removed uninstall registration key " + subkey);
        } else {
            append_log_line(
                layout.log_file,
                L"Could not remove uninstall registration key " + subkey + L" (error " + wide_from_utf8(std::to_string(delete_result)) + L")."
            );
        }
    }

    const LONG python_core_delete = RegDeleteTreeW(HKEY_CURRENT_USER, python_core_key.c_str());
    if (python_core_delete == ERROR_SUCCESS || python_core_delete == ERROR_FILE_NOT_FOUND) {
        append_log_line(layout.log_file, L"Removed PythonCore registration key " + python_core_key);
    } else {
        append_log_line(
            layout.log_file,
            L"Could not remove PythonCore registration key " + python_core_key + L" (error " + wide_from_utf8(std::to_string(python_core_delete)) + L")."
        );
    }

    for (const auto& cache_dir : bundle_cache_dirs) {
        if (cache_dir.empty()) {
            continue;
        }
        if (fs::exists(cache_dir, fs_error)) {
            fs::remove_all(cache_dir, fs_error);
            if (!fs_error) {
                append_log_line(layout.log_file, L"Removed cached installer bundle directory " + cache_dir.wstring());
            } else {
                append_log_line(layout.log_file, L"Could not remove cached installer bundle directory " + cache_dir.wstring());
                fs_error.clear();
            }
        }
    }
}

bool command_exists_on_path(const std::wstring& executable_name) {
    const DWORD required = SearchPathW(nullptr, executable_name.c_str(), nullptr, 0, nullptr, nullptr);
    return required > 0;
}

std::vector<std::wstring> enumerate_display_adapter_names() {
    std::vector<std::wstring> names;
    for (DWORD adapter_index = 0;; ++adapter_index) {
        DISPLAY_DEVICEW adapter{};
        adapter.cb = sizeof(adapter);
        if (!EnumDisplayDevicesW(nullptr, adapter_index, &adapter, 0)) {
            break;
        }
        std::wstring name = trim_copy(adapter.DeviceString);
        if (!name.empty()) {
            names.push_back(name);
        }
    }
    return names;
}

bool contains_case_insensitive(const std::wstring& text, const std::wstring& needle) {
    return lower_copy(text).find(lower_copy(needle)) != std::wstring::npos;
}

std::optional<std::wstring> env_value(const std::wstring& name) {
    const DWORD required = GetEnvironmentVariableW(name.c_str(), nullptr, 0);
    if (required == 0) {
        return std::nullopt;
    }
    std::wstring buffer(required, L'\0');
    GetEnvironmentVariableW(name.c_str(), buffer.data(), required);
    if (!buffer.empty() && buffer.back() == L'\0') {
        buffer.pop_back();
    }
    return trim_copy(buffer);
}

AcceleratorChoice detect_accelerator(const LauncherConfig& config, const ProjectLayout& layout) {
    const auto adapter_names = enumerate_display_adapter_names();
    for (const auto& name : adapter_names) {
        append_log_line(layout.log_file, L"Detected display adapter: " + name);
    }

    const bool has_nvidia_adapter = std::any_of(
        adapter_names.begin(),
        adapter_names.end(),
        [](const std::wstring& name) { return contains_case_insensitive(name, L"nvidia"); }
    );
    const bool has_nvidia_driver = has_nvidia_adapter || command_exists_on_path(L"nvidia-smi.exe")
        || fs::exists(LR"(C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe)");
    if (has_nvidia_driver) {
        AcceleratorChoice choice;
        choice.kind = AcceleratorKind::NvidiaCuda;
        choice.display_name = L"NVIDIA CUDA";
        choice.install_tag = L"nvidia-cu129";
        choice.index_url = config.torch_cuda_index_url;
        choice.note = L"NVIDIA driver detected. Installing the CUDA wheel set from the official cu129 index.";
        return choice;
    }

    const bool has_intel_arc = std::any_of(
        adapter_names.begin(),
        adapter_names.end(),
        [](const std::wstring& name) {
            return contains_case_insensitive(name, L"intel") && contains_case_insensitive(name, L"arc");
        }
    );
    if (has_intel_arc) {
        AcceleratorChoice choice;
        choice.kind = AcceleratorKind::IntelArcXpu;
        choice.display_name = L"Intel Arc / XPU";
        choice.install_tag = L"intel-xpu";
        choice.index_url = config.torch_xpu_index_url;
        choice.note = L"Intel Arc adapter detected. Installing the official PyTorch XPU wheels.";
        return choice;
    }

    const bool has_amd_adapter = std::any_of(
        adapter_names.begin(),
        adapter_names.end(),
        [](const std::wstring& name) {
            return contains_case_insensitive(name, L"amd") || contains_case_insensitive(name, L"radeon");
        }
    );
    const bool has_rocm_markers = env_value(L"ROCM_PATH").has_value()
        || env_value(L"HIP_PATH").has_value()
        || command_exists_on_path(L"rocminfo.exe")
        || command_exists_on_path(L"hipinfo.exe")
        || fs::exists(LR"(C:\Program Files\AMD\ROCm)")
        || fs::exists(LR"(C:\Program Files\AMD\HIP)");
    if (has_amd_adapter && has_rocm_markers) {
        AcceleratorChoice choice;
        choice.kind = AcceleratorKind::AmdRocmDetected;
        choice.display_name = L"AMD ROCm detected";
        choice.install_tag = L"amd-rocm-fallback-cpu";
        choice.note =
            L"AMD ROCm markers were detected, but the official native Windows PyTorch wheels currently do not provide a documented ROCm install path. Falling back to the CPU wheels for this launcher.";
        return choice;
    }

    AcceleratorChoice choice;
    choice.kind = AcceleratorKind::Cpu;
    choice.display_name = L"CPU";
    choice.install_tag = L"cpu";
    choice.note = L"No supported accelerator driver was detected. Installing the regular CPU wheels.";
    return choice;
}

std::wstring python_installer_url_for_version(const LauncherConfig& config, const std::wstring& version) {
    return replace_all(config.python_installer_url_template, L"{version}", version);
}

std::wstring python_embeddable_url_for_version(const LauncherConfig& config, const std::wstring& version) {
    return replace_all(config.python_embeddable_url_template, L"{version}", version);
}

fs::path windows_system_executable(const std::wstring& executable_name) {
    std::wstring buffer(MAX_PATH, L'\0');
    const UINT written = GetSystemDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
    fail_if(written == 0 || written >= buffer.size(), L"Could not determine the Windows system directory.");
    buffer.resize(written);
    return fs::path(buffer) / executable_name;
}

void extract_zip_archive(const fs::path& archive_file, const fs::path& destination_dir, const fs::path& log_file) {
    fail_if(!fs::exists(archive_file), L"ZIP archive not found: " + archive_file.wstring());
    fs::create_directories(destination_dir);

    const fs::path tar_path = windows_system_executable(L"tar.exe");
    if (fs::exists(tar_path)) {
        const int tar_exit = run_process(
            tar_path,
            {L"-xf", archive_file.wstring(), L"-C", destination_dir.wstring()},
            std::nullopt,
            log_file,
            true
        );
        if (tar_exit == 0) {
            return;
        }
        append_log_line(
            log_file,
            L"tar.exe failed to extract "
            + archive_file.wstring()
            + L" with exit code "
            + wide_from_utf8(std::to_string(tar_exit))
            + L". Falling back to PowerShell Expand-Archive."
        );
    }

    const auto powershell_path = find_executable_on_path(L"powershell.exe");
    fail_if(!powershell_path.has_value(), L"Neither tar.exe nor powershell.exe is available to unpack " + archive_file.wstring());
    const std::wstring command =
        L"Expand-Archive -LiteralPath "
        + quote_powershell_literal(archive_file.wstring())
        + L" -DestinationPath "
        + quote_powershell_literal(destination_dir.wstring())
        + L" -Force";
    const int powershell_exit = run_process(
        *powershell_path,
        {L"-NoProfile", L"-ExecutionPolicy", L"Bypass", L"-Command", command},
        std::nullopt,
        log_file,
        true
    );
    fail_if(
        powershell_exit != 0,
        L"Could not extract "
        + archive_file.wstring()
        + L" into "
        + destination_dir.wstring()
        + L" (PowerShell exit code "
        + wide_from_utf8(std::to_string(powershell_exit))
        + L")."
    );
}

void download_python_layout(const LauncherConfig& config, const ProjectLayout& layout) {
    std::error_code fs_error;
    fs::remove_all(layout.python_layout_dir, fs_error);
    fs::create_directories(layout.python_layout_dir);

    append_log_line(
        layout.log_file,
        L"Preparing standalone Python package layout in " + layout.python_layout_dir.wstring()
    );

    const std::vector<std::wstring> arguments = {
        L"/layout", layout.python_layout_dir.wstring(),
        L"/quiet",
        L"/log", layout.python_installer_log_file.wstring(),
        L"Include_core=1",
        L"Include_exe=1",
        L"Include_lib=1",
        L"Include_tcltk=1",
        L"Include_pip=0",
        L"Include_dev=0",
        L"Include_doc=0",
        L"Include_test=0",
        L"Include_symbols=0",
        L"Include_debug=0",
        L"Include_launcher=0",
        L"InstallLauncherAllUsers=0"
    };

    const int exit_code = run_process(
        layout.python_installer_file,
        arguments,
        std::nullopt,
        layout.log_file,
        true
    );
    if (exit_code != 0) {
        append_log_line(
            layout.log_file,
            L"Python layout download exited with code "
            + wide_from_utf8(std::to_string(exit_code))
            + L". Continuing to search any already cached package payloads."
        );
    }
}

std::optional<std::wstring> python_package_cache_version(const fs::path& candidate_path, const LauncherConfig& config) {
    static const std::wregex version_pattern(LR"(v(\d+\.\d+\.\d+\.\d+))", std::regex::icase);
    for (fs::path cursor = candidate_path.parent_path(); !cursor.empty(); cursor = cursor.parent_path()) {
        const std::wstring name = cursor.filename().wstring();
        std::wsmatch match;
        if (std::regex_search(name, match, version_pattern) && match.size() > 1) {
            const std::wstring version = match[1].str();
            if (version.rfind(config.python_series + L".", 0) == 0) {
                return version;
            }
        }
        if (cursor == cursor.root_path()) {
            break;
        }
    }
    return std::nullopt;
}

std::optional<fs::path> locate_python_runtime_msi(
    const LauncherConfig& config,
    const ProjectLayout& layout,
    const std::wstring& file_name
) {
    const fs::path direct_layout_candidate = layout.python_layout_dir / file_name;
    if (fs::exists(direct_layout_candidate)) {
        return direct_layout_candidate;
    }

    std::vector<fs::path> search_roots;
    if (const auto local_app_data = env_value(L"LOCALAPPDATA")) {
        search_roots.push_back(fs::path(*local_app_data) / "Package Cache");
    }
    if (const auto program_data = env_value(L"PROGRAMDATA")) {
        search_roots.push_back(fs::path(*program_data) / "Package Cache");
    }

    const std::wstring target_name = lower_copy(file_name);
    std::optional<fs::path> best_path;
    std::wstring best_version;

    for (const auto& root : search_roots) {
        if (!fs::exists(root)) {
            continue;
        }
        std::error_code iter_error;
        for (fs::recursive_directory_iterator it(
                 root,
                 fs::directory_options::skip_permission_denied,
                 iter_error);
             it != fs::recursive_directory_iterator();
             it.increment(iter_error)) {
            if (iter_error) {
                iter_error.clear();
                continue;
            }
            if (!it->is_regular_file(iter_error)) {
                if (iter_error) {
                    iter_error.clear();
                }
                continue;
            }
            const fs::path candidate = it->path();
            if (lower_copy(candidate.filename().wstring()) != target_name) {
                continue;
            }
            const auto version = python_package_cache_version(candidate, config);
            if (!version.has_value()) {
                continue;
            }
            if (!best_path.has_value() || version_less(best_version, *version)) {
                best_path = candidate;
                best_version = *version;
            }
        }
    }

    return best_path;
}

void extract_python_runtime_msi(const ProjectLayout& layout, const fs::path& msi_path) {
    const fs::path msiexec_path = windows_system_executable(L"msiexec.exe");
    append_log_line(
        layout.log_file,
        L"Extracting " + msi_path.filename().wstring() + L" into " + layout.python_base_dir.wstring()
    );
    const int exit_code = run_process(
        msiexec_path,
        {
            L"/a",
            msi_path.wstring(),
            L"/qn",
            L"TARGETDIR=" + layout.python_base_dir.wstring()
        },
        std::nullopt,
        layout.log_file,
        true
    );
    if (exit_code != 0) {
        throw make_error(
            L"Could not extract "
            + msi_path.filename().wstring()
            + L" into "
            + layout.python_base_dir.wstring()
            + L" (exit code "
            + wide_from_utf8(std::to_string(exit_code))
            + L")."
        );
    }
}

void write_embeddable_runtime_pth(const LauncherConfig& config, const ProjectLayout& layout) {
    std::wstring series_no_dot;
    series_no_dot.reserve(config.python_series.size());
    for (const wchar_t ch : config.python_series) {
        if (ch != L'.') {
            series_no_dot += ch;
        }
    }
    std::wstring project_root_entry;
    std::error_code relative_error;
    const fs::path relative_project_root = fs::relative(layout.project_root, layout.python_base_dir, relative_error);
    if (!relative_error && !relative_project_root.empty()) {
        project_root_entry = relative_project_root.wstring();
    } else {
        project_root_entry = layout.project_root.wstring();
    }
    const fs::path pth_file = layout.python_base_dir / (L"python" + series_no_dot + L"._pth");
    const std::wstring content_wide =
        std::wstring(L"python") + series_no_dot + L".zip\r\n"
        + L".\r\n"
        + L"DLLs\r\n"
        + L"Lib\r\n"
        + L"Lib\\site-packages\r\n"
        + project_root_entry + L"\r\n"
        + L"\r\n"
        + L"import site\r\n";
    const std::string content = utf8_from_wide(content_wide);
    write_text_file_utf8(pth_file, content);
}

void set_child_environment(const ProjectLayout& layout) {
    std::wstring path_prefix = (layout.python_env_dir / "Scripts").wstring() + L";" + layout.python_env_dir.wstring();
    const auto current_path = env_value(L"PATH").value_or(L"");
    if (!current_path.empty()) {
        path_prefix += L";" + current_path;
    }
    SetEnvironmentVariableW(L"PATH", path_prefix.c_str());
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"PIP_DISABLE_PIP_VERSION_CHECK", L"1");
    SetEnvironmentVariableW(L"PIP_CACHE_DIR", layout.pip_cache_dir.wstring().c_str());
}

bool python_supports_tkinter(const fs::path& python_exe, const fs::path& log_file) {
    if (!fs::exists(python_exe)) {
        return false;
    }
    const int exit_code = run_process(
        python_exe,
        {L"-c", L"import tkinter; import sys; sys.stdout.write('tk-ok')"},
        std::nullopt,
        log_file,
        true
    );
    return exit_code == 0;
}

bool python_supports_launcher_runtime(const fs::path& python_exe, const fs::path& log_file) {
    if (!fs::exists(python_exe)) {
        return false;
    }
    const int exit_code = run_process(
        python_exe,
        {L"-c", L"import ensurepip, tkinter, venv, sys; sys.stdout.write('launcher-runtime-ok')"},
        std::nullopt,
        log_file,
        true
    );
    return exit_code == 0;
}

bool python_has_importable_torch_stack(const fs::path& python_exe, const fs::path& log_file) {
    if (!fs::exists(python_exe)) {
        return false;
    }
    const int exit_code = run_process(
        python_exe,
        {L"-c", L"import torch, torchvision, torchaudio, sys; sys.stdout.write('torch-stack-ok')"},
        std::nullopt,
        log_file,
        true
    );
    return exit_code == 0;
}

void create_local_venv(const fs::path& base_python_exe, const ProjectLayout& layout) {
    if (fs::exists(layout.python_env_dir)) {
        fs::remove_all(layout.python_env_dir);
    }
    append_log_line(layout.log_file, L"Creating local virtual environment in " + layout.python_env_dir.wstring());
    const int exit_code = run_process(
        base_python_exe,
        {L"-m", L"venv", layout.python_env_dir.wstring(), L"--clear"},
        layout.project_root,
        layout.log_file,
        true
    );
    if (exit_code != 0) {
        throw make_error(
            L"Could not create the local Python environment (exit code "
            + wide_from_utf8(std::to_string(exit_code))
            + L")."
        );
    }
    fail_if(!fs::exists(layout.python_exe), L"Virtual environment was created, but python.exe is missing in " + layout.python_env_dir.wstring());
    fail_if(!fs::exists(layout.pythonw_exe), L"Virtual environment was created, but pythonw.exe is missing in " + layout.python_env_dir.wstring());
}

void repair_base_python(const ProjectLayout& layout) {
    append_log_line(layout.log_file, L"Attempting to repair the registered local Python install in " + layout.python_base_dir.wstring());
    const int exit_code = run_process(
        layout.python_installer_file,
        {
            L"/repair",
            L"/quiet",
            L"/log", layout.python_installer_log_file.wstring(),
            L"InstallAllUsers=0",
            L"InstallLauncherAllUsers=0",
            L"AssociateFiles=0",
            L"Include_launcher=0",
            L"PrependPath=0",
            L"Shortcuts=0"
        },
        std::nullopt,
        layout.log_file,
        true
    );
    if (exit_code != 0) {
        throw make_error(
            L"Python repair failed with exit code "
            + wide_from_utf8(std::to_string(exit_code))
            + L". See "
            + layout.python_installer_log_file.wstring()
        );
    }
}

void install_python_from_msi_payloads(const LauncherConfig& config, const ProjectLayout& layout, const std::wstring& python_version) {
    if (fs::exists(layout.python_base_dir)) {
        fs::remove_all(layout.python_base_dir);
    }
    fs::create_directories(layout.cache_dir);
    fs::create_directories(layout.logs_dir);
    fs::create_directories(layout.python_base_dir);

    const std::wstring installer_url = python_installer_url_for_version(config, python_version);
    download_file(installer_url, layout.python_installer_file, layout.log_file);

    if (fs::exists(layout.python_installer_log_file)) {
        fs::remove(layout.python_installer_log_file);
    }

    download_python_layout(config, layout);

    const std::vector<std::wstring> required_msi_names = {
        L"core.msi",
        L"exe.msi",
        L"lib.msi",
        L"tcltk.msi"
    };

    append_log_line(
        layout.log_file,
        L"Building the project-local Python " + python_version + L" runtime from standalone MSI payloads."
    );

    std::vector<fs::path> msi_paths;
    msi_paths.reserve(required_msi_names.size());
    for (const auto& msi_name : required_msi_names) {
        const auto located_msi = locate_python_runtime_msi(config, layout, msi_name);
        if (!located_msi.has_value()) {
            throw make_error(
                L"Could not locate the required Python package payload "
                + msi_name
                + L" for Python "
                + config.python_series
                + L". See "
                + layout.log_file.wstring()
            );
        }
        append_log_line(layout.log_file, L"Using " + msi_name + L" from " + located_msi->wstring());
        msi_paths.push_back(*located_msi);
    }

    for (const auto& msi_path : msi_paths) {
        extract_python_runtime_msi(layout, msi_path);
    }

    std::error_code fs_error;
    for (const auto& msi_name : required_msi_names) {
        fs::remove(layout.python_base_dir / msi_name, fs_error);
        fs_error.clear();
    }

    fail_if(
        !fs::exists(layout.python_base_exe) || !fs::exists(layout.python_basew_exe),
        L"Python runtime extraction finished, but python.exe or pythonw.exe is missing in " + layout.python_base_dir.wstring()
    );
    fail_if(
        !python_supports_launcher_runtime(layout.python_base_exe, layout.log_file),
        L"The extracted Python runtime is missing one or more required modules (tkinter, ensurepip, or venv)."
    );
}

void install_base_python(const LauncherConfig& config, const ProjectLayout& layout, const std::wstring& python_version) {
    if (fs::exists(layout.python_base_dir)) {
        fs::remove_all(layout.python_base_dir);
    }
    fs::create_directories(layout.cache_dir);
    fs::create_directories(layout.logs_dir);
    fs::create_directories(layout.python_base_dir);

    const std::wstring embeddable_url = python_embeddable_url_for_version(config, python_version);
    download_file(embeddable_url, layout.python_embeddable_file, layout.log_file);

    const std::wstring installer_url = python_installer_url_for_version(config, python_version);
    download_file(installer_url, layout.python_installer_file, layout.log_file);

    if (fs::exists(layout.python_installer_log_file)) {
        fs::remove(layout.python_installer_log_file);
    }

    append_log_line(
        layout.log_file,
        L"Building the project-local Python " + python_version + L" runtime from the embeddable package."
    );

    try {
        extract_zip_archive(layout.python_embeddable_file, layout.python_base_dir, layout.log_file);
        download_python_layout(config, layout);

        const std::vector<std::wstring> required_msi_names = {
            L"lib.msi",
            L"tcltk.msi"
        };
        for (const auto& msi_name : required_msi_names) {
            const auto located_msi = locate_python_runtime_msi(config, layout, msi_name);
            if (!located_msi.has_value()) {
                throw make_error(
                    L"Could not locate the required embeddable support payload "
                    + msi_name
                    + L" for Python "
                    + config.python_series
                    + L"."
                );
            }
            append_log_line(layout.log_file, L"Using " + msi_name + L" from " + located_msi->wstring());
            extract_python_runtime_msi(layout, *located_msi);
        }

        std::error_code fs_error;
        fs::remove(layout.python_base_dir / "lib.msi", fs_error);
        fs_error.clear();
        fs::remove(layout.python_base_dir / "tcltk.msi", fs_error);
        fs_error.clear();

        write_embeddable_runtime_pth(config, layout);

        fail_if(
            !fs::exists(layout.python_base_exe) || !fs::exists(layout.python_basew_exe),
            L"Embedded Python extraction finished, but python.exe or pythonw.exe is missing in " + layout.python_base_dir.wstring()
        );
        fail_if(
            !python_supports_launcher_runtime(layout.python_base_exe, layout.log_file),
            L"The embeddable Python runtime is missing one or more required modules (tkinter, ensurepip, or venv)."
        );
        append_log_line(layout.log_file, L"Embeddable Python runtime is ready.");
    } catch (const std::exception& exc) {
        append_log_line(
            layout.log_file,
            L"Embeddable runtime setup failed. Falling back to the MSI payload runtime builder."
        );
        append_log_line(layout.log_file, wide_from_utf8(exc.what()));
        install_python_from_msi_payloads(config, layout, python_version);
    }
}

void ensure_python_runtime(const LauncherConfig& config, const ProjectLayout& layout, const std::wstring& python_version) {
    const std::wstring installed_version = trim_copy(read_text_file_utf8(layout.python_version_file));
    const bool env_present = fs::exists(layout.python_exe) && fs::exists(layout.pythonw_exe);
    const bool base_present = fs::exists(layout.python_base_exe) && fs::exists(layout.python_basew_exe);
    std::wstring series_no_dot;
    for (const wchar_t ch : config.python_series) {
        if (ch != L'.') {
            series_no_dot += ch;
        }
    }
    const fs::path embeddable_pth_file = layout.python_base_dir / (L"python" + series_no_dot + L"._pth");
    if (base_present && fs::exists(embeddable_pth_file)) {
        write_embeddable_runtime_pth(config, layout);
    }
    const bool base_valid = base_present && python_supports_launcher_runtime(layout.python_base_exe, layout.log_file);
    if (env_present
        && base_valid
        && installed_version == python_version
        && python_supports_launcher_runtime(layout.python_exe, layout.log_file)) {
        append_log_line(layout.log_file, L"Local Python runtime " + python_version + L" is already present.");
        cleanup_local_python_registration(config, layout);
        return;
    }

    cleanup_local_python_registration(config, layout);

    if (!base_valid || installed_version != python_version) {
        install_base_python(config, layout, python_version);
    } else {
        append_log_line(layout.log_file, L"Reusing existing Python base install at " + layout.python_base_dir.wstring());
    }

    fail_if(
        !fs::exists(layout.python_base_exe) || !fs::exists(layout.python_basew_exe),
        L"Python installed, but python.exe was not found in " + layout.python_base_dir.wstring()
    );
    create_local_venv(layout.python_base_exe, layout);
    fail_if(
        !python_supports_launcher_runtime(layout.python_exe, layout.log_file),
        L"The local Python environment was created, but one or more required modules (tkinter, ensurepip, or venv) are unavailable."
    );
    write_text_file_utf8(layout.python_version_file, utf8_from_wide(python_version + L"\n"));
    cleanup_local_python_registration(config, layout);
}

void run_python(const ProjectLayout& layout, const std::vector<std::wstring>& arguments, const std::optional<fs::path>& cwd) {
    const int exit_code = run_process(layout.python_exe, arguments, cwd, layout.log_file, true);
    if (exit_code != 0) {
        throw make_error(L"Python command failed with exit code " + wide_from_utf8(std::to_string(exit_code)) + L". See " + layout.log_file.wstring());
    }
}

void ensure_packaging_tools(const ProjectLayout& layout) {
    append_log_line(layout.log_file, L"Bootstrapping pip, setuptools, and wheel.");
    run_python(layout, {L"-m", L"ensurepip", L"--upgrade"}, std::nullopt);
    run_python(
        layout,
        {
            L"-m", L"pip", L"install",
            L"--upgrade",
            L"--disable-pip-version-check",
            L"--no-warn-script-location",
            L"pip",
            L"setuptools",
            L"wheel"
        },
        std::nullopt
    );
}

std::wstring canonical_requirement_name(const std::wstring& requirement_line) {
    const std::wstring trimmed = trim_copy(requirement_line);
    if (trimmed.empty() || trimmed.front() == L'#' || trimmed.front() == L'-') {
        return L"";
    }
    size_t end = 0;
    while (end < trimmed.size()) {
        const wchar_t ch = trimmed[end];
        if (ch == L' ' || ch == L'\t' || ch == L'[' || ch == L';' || ch == L'@' || ch == L'=' || ch == L'<' || ch == L'>' || ch == L'!' || ch == L'~') {
            break;
        }
        ++end;
    }
    std::wstring name = lower_copy(trimmed.substr(0, end));
    std::wstring normalized;
    normalized.reserve(name.size());
    for (const wchar_t ch : name) {
        if (ch != L'-' && ch != L'_' && ch != L'.') {
            normalized += ch;
        }
    }
    return normalized;
}

std::optional<fs::path> prepare_filtered_requirements(const ProjectLayout& layout) {
    if (!fs::exists(layout.requirements_file)) {
        append_log_line(layout.log_file, L"No requirements.txt was found. Skipping application requirements sync.");
        return std::nullopt;
    }

    std::wifstream input(layout.requirements_file);
    if (!input) {
        throw make_error(L"Could not open " + layout.requirements_file.wstring() + L" for reading.");
    }
    input.imbue(std::locale::classic());

    std::wstring output;
    bool has_package_lines = false;
    std::wstring line;
    while (std::getline(input, line)) {
        const std::wstring package_name = canonical_requirement_name(line);
        if (package_name == L"torch" || package_name == L"torchvision" || package_name == L"torchaudio") {
            append_log_line(layout.log_file, L"Skipping torch-related requirement line: " + trim_copy(line));
            continue;
        }
        const std::wstring trimmed = trim_copy(line);
        if (!trimmed.empty() && trimmed.front() != L'#') {
            has_package_lines = true;
        }
        output += line;
        output += L"\r\n";
    }

    write_text_file_utf8(layout.filtered_requirements_file, utf8_from_wide(output));
    if (!has_package_lines) {
        append_log_line(layout.log_file, L"requirements.txt contains no non-torch packages after filtering.");
        return std::nullopt;
    }
    return layout.filtered_requirements_file;
}

std::vector<std::wstring> pip_install_arguments(
    const std::vector<std::wstring>& packages,
    const std::optional<std::wstring>& index_url
) {
    std::vector<std::wstring> arguments = {
        L"-m", L"pip", L"install",
        L"--upgrade",
        L"--disable-pip-version-check",
        L"--no-warn-script-location",
        L"--upgrade-strategy", L"only-if-needed"
    };
    if (index_url.has_value()) {
        arguments.push_back(L"--index-url");
        arguments.push_back(*index_url);
    }
    arguments.insert(arguments.end(), packages.begin(), packages.end());
    return arguments;
}

void uninstall_torch_packages(const ProjectLayout& layout) {
    append_log_line(layout.log_file, L"Removing previously installed torch packages so the runtime can switch variants cleanly.");
    const int exit_code = run_process(
        layout.python_exe,
        {
            L"-m", L"pip", L"uninstall",
            L"-y",
            L"torch",
            L"torchvision",
            L"torchaudio"
        },
        std::nullopt,
        layout.log_file,
        true
    );
    if (exit_code != 0) {
        append_log_line(layout.log_file, L"pip uninstall returned exit code " + wide_from_utf8(std::to_string(exit_code)) + L". Continuing.");
    }
}

void install_torch_stack(const ProjectLayout& layout, const AcceleratorChoice& accelerator) {
    const std::wstring previous_target = trim_copy(read_text_file_utf8(layout.torch_target_file));
    if (previous_target == accelerator.install_tag && python_has_importable_torch_stack(layout.python_exe, layout.log_file)) {
        append_log_line(
            layout.log_file,
            L"Torch stack for " + accelerator.display_name + L" is already installed and importable. Skipping reinstall."
        );
        return;
    }
    if (!previous_target.empty() && previous_target != accelerator.install_tag) {
        uninstall_torch_packages(layout);
    }

    const auto install_packages = [&](const std::optional<std::wstring>& index_url, const std::wstring& description) {
        append_log_line(layout.log_file, description);
        run_python(
            layout,
            pip_install_arguments({L"torch", L"torchvision", L"torchaudio"}, index_url),
            std::nullopt
        );
    };

    if (accelerator.kind == AcceleratorKind::AmdRocmDetected) {
        append_log_line(layout.log_file, accelerator.note);
        install_packages(std::nullopt, L"Installing the regular CPU torch stack.");
        write_text_file_utf8(layout.torch_target_file, utf8_from_wide(accelerator.install_tag + L"\n"));
        return;
    }

    try {
        if (accelerator.index_url.has_value()) {
            install_packages(accelerator.index_url, L"Installing the " + accelerator.display_name + L" torch stack.");
        } else {
            install_packages(std::nullopt, L"Installing the regular CPU torch stack.");
        }
        write_text_file_utf8(layout.torch_target_file, utf8_from_wide(accelerator.install_tag + L"\n"));
    } catch (const std::exception& exc) {
        if (accelerator.kind == AcceleratorKind::Cpu) {
            throw;
        }
        append_log_line(layout.log_file, L"Accelerator-specific torch install failed. Falling back to the regular CPU wheels.");
        append_log_line(layout.log_file, wide_from_utf8(exc.what()));
        uninstall_torch_packages(layout);
        install_packages(std::nullopt, L"Installing CPU torch wheels after fallback.");
        write_text_file_utf8(layout.torch_target_file, utf8_from_wide(L"cpu-fallback\n"));
    }
}

void install_project_requirements(const ProjectLayout& layout) {
    const auto filtered = prepare_filtered_requirements(layout);
    if (!filtered.has_value()) {
        return;
    }
    append_log_line(layout.log_file, L"Installing filtered application requirements from " + filtered->wstring());
    run_python(
        layout,
        {
            L"-m", L"pip", L"install",
            L"--upgrade",
            L"--disable-pip-version-check",
            L"--no-warn-script-location",
            L"--upgrade-strategy", L"only-if-needed",
            L"-r", filtered->wstring()
        },
        layout.project_root
    );
}

void launch_run_script(const ProjectLayout& layout, const std::vector<std::wstring>& extra_args) {
    std::vector<std::wstring> arguments;
    arguments.push_back(layout.run_script.wstring());
    arguments.insert(arguments.end(), extra_args.begin(), extra_args.end());

    const fs::path preferred_python = fs::exists(layout.pythonw_exe) ? layout.pythonw_exe : layout.python_exe;
    append_log_line(layout.log_file, L"Launching " + layout.run_script.wstring() + L" with " + preferred_python.wstring());
    start_detached_process(preferred_python, arguments, layout.project_root, true);
}

ProjectLayout resolve_project_layout(const LauncherConfig& config) {
    ProjectLayout layout{};
    const fs::path exe_path = executable_path();
    layout.launcher_dir = exe_path.parent_path();

    fs::path cursor = layout.launcher_dir;
    while (true) {
        if (fs::exists(cursor / "run.py")) {
    layout.project_root = cursor;
            break;
        }
        if (cursor == cursor.root_path()) {
            throw make_error(L"Could not find run.py. Place launch.exe in the project root or a subfolder below it.");
        }
        cursor = cursor.parent_path();
    }

    layout.runtime_dir = layout.project_root / config.runtime_dir_name;
    layout.cache_dir = layout.runtime_dir / config.cache_dir_name;
    layout.python_layout_dir = layout.cache_dir / "python-layout";
    layout.logs_dir = layout.runtime_dir / config.logs_dir_name;
    layout.log_file = layout.logs_dir / "launcher.log";
    layout.python_base_dir = layout.runtime_dir / config.python_base_dir_name;
    layout.python_env_dir = layout.runtime_dir / config.python_env_dir_name;
    layout.python_base_exe = layout.python_base_dir / "python.exe";
    layout.python_basew_exe = layout.python_base_dir / "pythonw.exe";
    layout.python_exe = layout.python_env_dir / "Scripts" / "python.exe";
    layout.pythonw_exe = layout.python_env_dir / "Scripts" / "pythonw.exe";
    layout.python_version_file = layout.runtime_dir / config.python_version_file_name;
    layout.torch_target_file = layout.runtime_dir / config.torch_target_file_name;
    layout.filtered_requirements_file = layout.cache_dir / config.filtered_requirements_file_name;
    layout.python_index_cache_file = layout.cache_dir / "python-ftp-index.html";
    layout.python_installer_file = layout.cache_dir / "python-runtime-installer.exe";
    layout.python_embeddable_file = layout.cache_dir / "python-runtime-embed.zip";
    layout.python_installer_log_file = layout.logs_dir / "python-installer.log";
    layout.requirements_file = layout.project_root / "requirements.txt";
    layout.run_script = layout.project_root / "run.py";
    layout.pip_cache_dir = layout.cache_dir / "pip";

    fs::create_directories(layout.runtime_dir);
    fs::create_directories(layout.cache_dir);
    fs::create_directories(layout.logs_dir);
    fs::create_directories(layout.pip_cache_dir);
    return layout;
}

void bootstrap_worker(WorkerContext context) {
    try {
        append_log_line(context.layout.log_file, L"Launcher starting in " + context.layout.project_root.wstring());
        post_status(context.hwnd, 1, L"Project discovered. Checking Python 3.12...");
        post_log(context.hwnd, L"Project root: " + context.layout.project_root.wstring());

        const std::wstring python_version = resolve_latest_python_version(context.config, context.layout);
        post_log(context.hwnd, L"Using Python " + python_version);

        post_status(context.hwnd, 2, L"Ensuring local Python runtime...");
        ensure_python_runtime(context.config, context.layout, python_version);

        set_child_environment(context.layout);
        post_status(context.hwnd, 3, L"Bootstrapping pip tools...");
        ensure_packaging_tools(context.layout);

        post_status(context.hwnd, 4, L"Detecting accelerator runtime...");
        const AcceleratorChoice accelerator = detect_accelerator(context.config, context.layout);
        post_log(context.hwnd, accelerator.note);

        post_status(context.hwnd, 5, L"Installing torch, torchvision, and torchaudio...");
        install_torch_stack(context.layout, accelerator);

        post_status(context.hwnd, 6, L"Installing filtered project requirements...");
        install_project_requirements(context.layout);

        post_status(context.hwnd, 7, L"Launching run.py...");
        launch_run_script(context.layout, context.extra_args);

        post_launched(context.hwnd, L"run.py started successfully. Closing launcher...");
    } catch (const std::exception& exc) {
        const std::wstring message = wide_from_utf8(exc.what());
        append_log_line(context.layout.log_file, L"Bootstrap failed: " + message);
        post_failed(context.hwnd, message + L"\r\n\r\nSee the log at:\r\n" + context.layout.log_file.wstring());
    }
}

void append_log_to_edit(HWND edit_control, const std::wstring& text) {
    const int current_length = GetWindowTextLengthW(edit_control);
    SendMessageW(edit_control, EM_SETSEL, current_length, current_length);
    std::wstring line = text + L"\r\n";
    SendMessageW(edit_control, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(line.c_str()));
    SendMessageW(edit_control, EM_SCROLLCARET, 0, 0);
}

LRESULT CALLBACK launcher_window_proc(HWND hwnd, UINT message, WPARAM w_param, LPARAM l_param) {
    auto* state = reinterpret_cast<WindowState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (message) {
    case WM_CREATE: {
        auto* new_state = new WindowState();
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(new_state));
        state = new_state;

        state->status_label = CreateWindowExW(
            0,
            WC_STATICW,
            L"Preparing launcher...",
            WS_CHILD | WS_VISIBLE,
            18,
            18,
            780,
            26,
            hwnd,
            nullptr,
            nullptr,
            nullptr
        );

        state->progress_bar = CreateWindowExW(
            0,
            PROGRESS_CLASSW,
            nullptr,
            WS_CHILD | WS_VISIBLE,
            18,
            54,
            780,
            22,
            hwnd,
            nullptr,
            nullptr,
            nullptr
        );
        SendMessageW(state->progress_bar, PBM_SETRANGE32, 0, kProgressMax);
        SendMessageW(state->progress_bar, PBM_SETPOS, 0, 0);

        state->detail_label = CreateWindowExW(
            0,
            WC_STATICW,
            L"Logs will appear below.",
            WS_CHILD | WS_VISIBLE,
            18,
            84,
            780,
            18,
            hwnd,
            nullptr,
            nullptr,
            nullptr
        );

        state->log_edit = CreateWindowExW(
            WS_EX_CLIENTEDGE,
            WC_EDITW,
            nullptr,
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
            18,
            112,
            780,
            362,
            hwnd,
            nullptr,
            nullptr,
            nullptr
        );

        state->close_button = CreateWindowExW(
            0,
            WC_BUTTONW,
            L"Close",
            WS_CHILD | WS_VISIBLE | WS_DISABLED | BS_PUSHBUTTON,
            708,
            486,
            90,
            30,
            hwnd,
            reinterpret_cast<HMENU>(1),
            nullptr,
            nullptr
        );
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(w_param) == 1 && state && state->can_close) {
            DestroyWindow(hwnd);
            return 0;
        }
        break;
    case WM_CLOSE:
        if (state && !state->can_close) {
            MessageBeep(MB_ICONINFORMATION);
            return 0;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        delete state;
        PostQuitMessage(0);
        return 0;
    case WMU_STATUS:
    case WMU_LOG:
    case WMU_FAILED:
    case WMU_LAUNCHED: {
        std::unique_ptr<UiMessage> payload(reinterpret_cast<UiMessage*>(l_param));
        if (!state || !payload) {
            return 0;
        }

        if (message == WMU_STATUS) {
            SetWindowTextW(state->status_label, payload->text.c_str());
            SendMessageW(state->progress_bar, PBM_SETPOS, payload->step, 0);
        } else if (message == WMU_LOG) {
            append_log_to_edit(state->log_edit, payload->text);
        } else if (message == WMU_FAILED) {
            SetWindowTextW(state->status_label, L"Launcher failed.");
            SendMessageW(state->progress_bar, PBM_SETPOS, kProgressMax, 0);
            append_log_to_edit(state->log_edit, payload->text);
            state->can_close = true;
            EnableWindow(state->close_button, TRUE);
        } else if (message == WMU_LAUNCHED) {
            SetWindowTextW(state->status_label, payload->text.c_str());
            SendMessageW(state->progress_bar, PBM_SETPOS, kProgressMax, 0);
            append_log_to_edit(state->log_edit, payload->text);
            state->can_close = true;
            EnableWindow(state->close_button, TRUE);
            PostMessageW(hwnd, WM_CLOSE, 0, 0);
        }
        return 0;
    }
    default:
        break;
    }

    return DefWindowProcW(hwnd, message, w_param, l_param);
}

HWND create_launcher_window(HINSTANCE instance, const fs::path& log_file) {
    INITCOMMONCONTROLSEX controls{};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&controls);

    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.lpfnWndProc = launcher_window_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = kWindowClassName;
    window_class.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    window_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassExW(&window_class);

    HWND hwnd = CreateWindowExW(
        0,
        kWindowClassName,
        L"Local Python Runtime Launcher",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        836,
        572,
        nullptr,
        nullptr,
        instance,
        nullptr
    );
    fail_if(hwnd == nullptr, L"Could not create the launcher window: " + last_error_message(GetLastError()));

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    if (auto* window_state = reinterpret_cast<WindowState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA))) {
        window_state->log_file = log_file;
        SetWindowTextW(window_state->detail_label, log_file.wstring().c_str());
    }
    return hwnd;
}

std::vector<std::wstring> parse_extra_arguments() {
    int argument_count = 0;
    LPWSTR* arguments = CommandLineToArgvW(GetCommandLineW(), &argument_count);
    if (!arguments) {
        return {};
    }
    std::vector<std::wstring> result;
    for (int index = 1; index < argument_count; ++index) {
        result.emplace_back(arguments[index]);
    }
    LocalFree(arguments);
    return result;
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    try {
        LauncherConfig config{};
        ProjectLayout layout = resolve_project_layout(config);
        const HWND hwnd = create_launcher_window(instance, layout.log_file);

        post_log(hwnd, L"Launcher executable: " + executable_path().wstring());
        post_log(hwnd, L"Project root: " + layout.project_root.wstring());
        if (fs::exists(layout.requirements_file)) {
            post_log(hwnd, L"Using requirements file: " + layout.requirements_file.wstring());
        } else {
            post_log(hwnd, L"No requirements.txt was found next to run.py.");
        }

        WorkerContext context{};
        context.config = config;
        context.layout = layout;
        context.hwnd = hwnd;
        context.extra_args = parse_extra_arguments();

        std::thread worker([context]() mutable { bootstrap_worker(std::move(context)); });
        worker.detach();

        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        return 0;
    } catch (const std::exception& exc) {
        MessageBoxW(
            nullptr,
            wide_from_utf8(exc.what()).c_str(),
            L"Launcher startup failed",
            MB_OK | MB_ICONERROR
        );
        return 1;
    }
}
