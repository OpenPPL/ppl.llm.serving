#ifndef __PPL_LLM_SERVING_BACKTRACE_H__
#define __PPL_LLM_SERVING_BACKTRACE_H__

#include <string>
#include <iostream>
#include <string.h>
#include <execinfo.h>
#include <unistd.h>
#include <cxxabi.h>

namespace ppl { namespace llm {

class BackTrace final {
public:
    static std::string Get() {
        const uint32_t MAX_STACK_SIZE = 128;
        void* stk[MAX_STACK_SIZE];
        size_t stk_size;

        stk_size = backtrace(stk, MAX_STACK_SIZE);
        if (stk_size == 0) {
            return std::string();
        }

        char** strings = backtrace_symbols(stk, stk_size);

        std::string ret;
        for (size_t i = 0; i < stk_size; ++i) {
            auto addr = ExtractAddr(strings[i]);
            if (addr.empty()) {
                continue;
            }
            auto cmd = "addr2line -fp -e " + GetBinPath() + " " + addr;
            auto line = ExecuteCmd(cmd);
            if (line.empty()) {
                continue;
            }

            std::string func, pos;
            ExtractFuncAndPos(line, &func, &pos);
            char tmp[64] = {'\0'};
            sprintf(tmp, "#%ld\t", i);
            ret += tmp + func + " at " + pos + "\n";
        }
        ret += "\n";
        ::free(strings);

        return ret;
    }

private:
    static std::string ExtractAddr(const char* content) {
        char tmp[128] = {'\0'}, tmp2[128] = {'\0'};
        char binname[128] = {'\0'};
        uint64_t addr;
        sscanf(content, "%s%s", binname, tmp);
        sscanf(tmp, "%3c%lx", tmp2, &addr); // tmp -> [0x1234567]
        int len = sprintf(tmp, "0x%lx", addr);
        return std::string(tmp, len);
    }

    static std::string ExecuteCmd(const std::string& cmd) {
        auto fp = popen(cmd.c_str(), "r");
        if (!fp) {
            std::cerr << "exec cmd [" << cmd << "] failed." << std::endl;
            return std::string();
        }

        char buf[1024] = {'\0'};
        auto unused = fgets(buf, 1024, fp);
        (void)unused;
        int len = strlen(buf);
        fclose(fp);
        return std::string(buf, len - 1); // remove trailing '\n'
    }

    static std::string GetBinPath() {
        char tmp[256] = {'\0'};
        sprintf(tmp, "%d", getpid());
        auto exe_info = "/proc/" + std::string(tmp) + "/exe";
        auto len = readlink(exe_info.c_str(), tmp, 256);
        if (len <= 0) {
            std::cerr << "GetBinPath() failed." << std::endl;
            return std::string();
        }
        return std::string(tmp, len);
    }

    static std::string Demangle(const std::string& name) {
        std::string res;
        size_t size = 0;
        int status = 0;
        auto ret = abi::__cxa_demangle(name.c_str(), nullptr, &size, &status);
        if (ret) {
            res.assign(ret);
            free(ret);
        } else {
            /*
              status:
                0: The demangling operation succeeded.
                -1: A memory allocation failiure occurred.
                -2: mangled_name is not a valid name under the C++ ABI mangling rules.
                -3: One of the arguments is invalid.
            */
            //std::cerr << "demangle failed, status = " << status << ", name -> " << name << std::endl;
            res = name;
        }

        return res;
    }

    static void ExtractFuncAndPos(const std::string& line, std::string* func,
                                  std::string* pos) {
        char tmp[64] = {'\0'};
        char func_name[512] = {'\0'};
        char pos_buf[1024] = {'\0'};
        sscanf(line.c_str(), "%s%s%s", func_name, tmp, pos_buf);
        func->assign(Demangle(func_name));
        pos->assign(pos_buf);
    }
};

}}

#endif
