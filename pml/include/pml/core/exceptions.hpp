#include <new>
#include <source_location>
#include <string>

namespace pml {

class custom_bad_alloc : public std::bad_alloc {
  private:
    std::string message;

  public:
    explicit custom_bad_alloc(const std::string& descriprion,
                              const std::source_location& loc) noexcept {
        message = std::string(loc.file_name()) + ": " + std::to_string(loc.line()) + " [" +
                  loc.function_name() + "] " + descriprion;
    }

    const char* what() const noexcept override {
        return message.c_str();
    }
};

} // namespace pml
