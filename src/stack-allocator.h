#pragma once

#include "core/slang-basic.h"
#include "slang-rhi.h"

#include <stdexcept>
#include <vector>

// Represents a sub-allocation view
struct BufferView
{
    rhi::IBuffer* buffer = nullptr;
    size_t offset = 0;
    size_t size = 0;

    BufferView() = default;
    BufferView(rhi::IBuffer* buf, size_t off, size_t sz)
        : buffer(buf), offset(off), size(sz)
    {
        if (!buffer && size > 0)
            throw std::runtime_error("BufferView: buffer is null but size > 0");
    }
    BufferView(rhi::IBuffer* buf)
        : buffer(buf), offset(0), size(buf->getDesc().size)
    {
    }
    BufferView(Slang::ComPtr<rhi::IBuffer> buf)
        : buffer(buf.get()), offset(0), size(buf->getDesc().size)
    {
    }

    BufferView tail(size_t tailOffset) const
    {
        if (tailOffset > size)
            throw std::runtime_error("BufferView::tail: tailOffset exceeds size");
        return BufferView(buffer, offset + tailOffset, size - tailOffset);
    }

    BufferView head(size_t headSize) const
    {
        if (headSize > size)
            throw std::runtime_error("BufferView::head: headSize exceeds size");
        return BufferView(buffer, offset, headSize);
    }

    // Helper to get GPU address if needed
    uint64_t getDeviceAddress() const { return buffer->getDeviceAddress() + offset; }

    explicit operator bool() const { return buffer != nullptr && size > 0; }
};

class InferencingContext;

class StackAllocator : public Slang::RefObject
{
public:
    struct State
    {
        size_t pageIndex;
        size_t offset;
    };

private:
    InferencingContext* context;
    size_t defaultPageSize;

    struct Page
    {
        Slang::ComPtr<rhi::IBuffer> buffer;
        size_t capacity;
        size_t currentOffset;
    };

    // All pages created so far. We reuse them instead of destroying them.
    std::vector<Page> pages;

    // The index of the page we are currently writing to
    size_t currentPageIndex = 0;

    // Stack for push/pop
    std::vector<State> stateStack;

    // Helper to add a new page
    void addPage(size_t minSize);

public:
    StackAllocator(
        InferencingContext* ctx,
        size_t defaultPageSize = 1024 * 1024 * 64); // 64MB Default

    // Main API
    BufferView allocate(size_t size, size_t alignment = 256);

    void push();
    void pop();

    // Clear everything (e.g., at end of frame)
    void reset();
};