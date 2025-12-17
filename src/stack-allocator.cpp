#include "stack-allocator.h"

#include "inference-context.h"

#include <algorithm>

StackAllocator::StackAllocator(InferencingContext* ctx, size_t defaultPageSize)
    : context(ctx), defaultPageSize(defaultPageSize)
{
    // Create the first page immediately
    addPage(defaultPageSize);
}

void StackAllocator::addPage(size_t minSize)
{
    size_t size = std::max(defaultPageSize, minSize);

    // Create GPU buffer
    auto buf = context->createPersistentBuffer(nullptr, size, "StackAllocator_Page");

    Page p;
    p.buffer = buf;
    p.capacity = size;
    p.currentOffset = 0;

    pages.push_back(p);
}

BufferView StackAllocator::allocate(size_t size, size_t alignment)
{
    Page* current = &pages[currentPageIndex];

    // 1. Align the offset
    size_t alignedOffset = (current->currentOffset + alignment - 1) & ~(alignment - 1);

    // 2. Check if it fits in current page
    if (alignedOffset + size > current->capacity)
    {
        // Does NOT fit. Move to next page.

        // Is there a next page available in our pool?
        if (currentPageIndex + 1 < pages.size())
        {
            currentPageIndex++;
            current = &pages[currentPageIndex];

            // Check if existing next page is big enough (edge case: huge allocation)
            if (size > current->capacity)
            {
                // This reused page is too small. We technically could replace it,
                // but for simplicity, let's just create a new specific huge page
                // and insert it here, or just fail for now.
                // Better strategy: Just append a new huge page at the end and swap?
                throw std::runtime_error("StackAllocator: Allocation size exceeds page capacity.");
            }

            current->currentOffset = 0; // Reset the reused page
            alignedOffset = 0;          // New page starts at 0
        }
        else
        {
            // No next page. Create one.
            addPage(size);
            currentPageIndex++;
            current = &pages[currentPageIndex];
            alignedOffset = 0;
        }
    }

    // 3. Commit allocation
    current->currentOffset = alignedOffset + size;

    return BufferView{current->buffer.get(), alignedOffset, size};
}

void StackAllocator::push()
{
    // Save exactly where we are
    State s;
    s.pageIndex = currentPageIndex;
    s.offset = pages[currentPageIndex].currentOffset;
    stateStack.push_back(s);
}

void StackAllocator::pop()
{
    if (stateStack.empty())
        throw std::runtime_error("StackAllocator: Pop without Push");

    State s = stateStack.back();
    stateStack.pop_back();

    // Restore State
    currentPageIndex = s.pageIndex;
    pages[currentPageIndex].currentOffset = s.offset;

    // Note: We do NOT destroy pages > currentPageIndex.
    // We keep them in 'pages' vector to reuse them in the next alloc() sequence.
    // This is the "Pool" behavior.
}

void StackAllocator::reset()
{
    stateStack.clear();
    currentPageIndex = 0;
    pages[0].currentOffset = 0;
    // Again, keep all allocated pages alive for next frame.
}